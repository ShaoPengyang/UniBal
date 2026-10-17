"""UniBal: BalDRO-DV forget loss and prototype-guided primal-dual retention."""

import math
import traceback

import torch
from torch.utils.data import DataLoader, Subset
from transformers import TrainerCallback

from trainer.unlearn.npo import DrNPO
from trainer.unlearn.prototypes import load_or_build_prototypes
from trainer.unlearn.satimp import DrSatImp
from trainer.unlearn.simnpo import DrSimNPO
from trainer.utils import compute_batch_nll


class _DualUpdateCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        # HF 4.45 callbacks receive the unwrapped model even under ZeRO/FSDP.
        self.trainer.update_dual(self.trainer.model_wrapped)
        return control


class _UniBalMixin:
    def __init__(
        self,
        semantic_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        n_prototypes=40,
        dual_step_size=0.05,
        retain_loss_eps=0.16,
        initial_lambda=1.0,
        probe_batch_size=8,
        encoder_batch_size=32,
        encoder_max_length=256,
        medoid_sample_size=5000,
        probe_cache_path=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.retain_loss_type != "NLL" or self.retain_dro:
            raise ValueError("UniBal requires a standard NLL retain constraint")
        if dual_step_size <= 0 or retain_loss_eps < 0 or initial_lambda < 0:
            raise ValueError("Invalid primal-dual hyperparameters")
        if probe_batch_size <= 0:
            raise ValueError("probe_batch_size must be positive")
        if (
            not hasattr(self.train_dataset, "retain")
            or self.train_dataset.retain is None
        ):
            raise ValueError("UniBal requires a ForgetRetainDataset with retain data")
        self.dual_step_size = dual_step_size
        self.retain_loss_eps = retain_loss_eps
        self.dual_lambda = float(initial_lambda)
        self.alpha = self.dual_lambda
        self.probe_batch_size = probe_batch_size
        retain = self.train_dataset.retain
        cache_path = (
            probe_cache_path or f"{self.args.output_dir}/retain_prototypes.json"
        )
        payload = [None]
        if self.accelerator.is_main_process:
            try:
                payload[0] = load_or_build_prototypes(
                    retain,
                    self.tokenizer,
                    semantic_encoder_name,
                    n_prototypes,
                    cache_path,
                    encoder_batch_size,
                    encoder_max_length,
                    medoid_sample_size,
                    self.args.seed,
                )
            except Exception:
                payload[0] = {"error": traceback.format_exc()}
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(payload, src=0)
        if isinstance(payload[0], dict) and "error" in payload[0]:
            raise RuntimeError(
                f"UniBal prototype creation failed: {payload[0]['error']}"
            )
        self.probe_indices, self.probe_weights = payload[0]
        if (
            len(self.probe_indices) != len(self.probe_weights)
            or not self.probe_indices
            or any(
                not math.isfinite(weight) or weight <= 0
                for weight in self.probe_weights
            )
            or abs(sum(self.probe_weights) - 1.0) > 1e-5
        ):
            raise ValueError("Invalid UniBal prototype indices or weights")
        self.add_callback(_DualUpdateCallback(self))

    def compute_retain_loss(self, model, retain_inputs):
        """Match the paper's mean per-example next-token loss."""
        losses, _ = compute_batch_nll(model, retain_inputs)
        token_counts = retain_inputs["labels"][:, 1:].ne(-100).sum(-1).clamp_min(1)
        return (losses.float() / token_counts).mean()

    def update_dual(self, model):
        retain = self.train_dataset.retain
        loader = DataLoader(
            Subset(retain, self.probe_indices),
            batch_size=self.probe_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )
        previous_mode = model.training
        weighted_sum = 0.0
        offset = 0
        try:
            model.eval()
            with torch.inference_mode():
                for batch in loader:
                    batch = self._prepare_inputs(batch)
                    inputs = {
                        key: batch[key]
                        for key in ("input_ids", "attention_mask", "labels")
                    }
                    with self.compute_loss_context_manager():
                        losses, _ = compute_batch_nll(model, inputs)
                    counts = inputs["labels"][:, 1:].ne(-100).sum(-1).clamp_min(1)
                    per_example = losses.float() / counts
                    count = len(per_example)
                    weights = per_example.new_tensor(
                        self.probe_weights[offset : offset + count]
                    )
                    weighted_sum += float((per_example * weights).sum().item())
                    offset += count
        finally:
            model.train(previous_mode)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            value = torch.tensor(weighted_sum, device=self.args.device)
            torch.distributed.all_reduce(value)
            weighted_sum = (value / torch.distributed.get_world_size()).item()
        self.dual_lambda = max(
            0.0,
            self.dual_lambda
            + self.dual_step_size * (weighted_sum - self.retain_loss_eps),
        )
        self.alpha = self.dual_lambda
        self.log({"probe_retain_loss": weighted_sum, "dual_lambda": self.dual_lambda})


class UniBalNPO(_UniBalMixin, DrNPO):
    pass


class UniBalSimNPO(_UniBalMixin, DrSimNPO):
    pass


class UniBalSatImp(_UniBalMixin, DrSatImp):
    pass
