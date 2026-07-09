import json
import logging
import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import TrainerCallback

from trainer.unlearn.npo import NPO
from trainer.unlearn.simnpo import SimNPO
from trainer.utils import compute_batch_nll

logger = logging.getLogger(__name__)


class ProbeRetainLambdaMixin:
    """Dynamically update retain lambda (alpha) using probe retain loss."""

    def __init__(
        self,
        retain_loss_eps=0.0,
        dual_step_size=1.0,
        dual_warmup_epochs=0,
        probe_budget_k=None,
        probe_build_batch_size=None,
        probe_eval_batch_size=16,
        probe_eval_interval=1,
        probe_cache_file=None,
        probe_selection_method="random",
        probe_cluster_max_iter=20,
        probe_seed=42,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not hasattr(self, "alpha"):
            raise AttributeError("ProbeRetainLambdaMixin requires trainer to define `alpha`.")
        if not hasattr(self, "train_dataset") or not hasattr(self.train_dataset, "retain"):
            raise AttributeError(
                "ProbeRetainLambdaMixin requires `train_dataset.retain` to exist."
            )

        self.retain_loss_eps = float(retain_loss_eps)
        self.dual_step_size = float(dual_step_size)
        self.probe_build_batch_size = int(probe_build_batch_size or probe_eval_batch_size)
        self.probe_eval_batch_size = int(probe_eval_batch_size)
        self.probe_eval_interval = max(1, int(probe_eval_interval))
        self.probe_selection_method = str(probe_selection_method).lower()
        self.probe_cluster_max_iter = max(1, int(probe_cluster_max_iter))
        self.probe_seed = int(probe_seed)

        self.full_retain_dataset = self.train_dataset.retain
        if probe_budget_k is None:
            probe_budget_k = len(self.full_retain_dataset)
        self.probe_budget_k = max(1, int(probe_budget_k))

        if probe_cache_file is None:
            probe_cache_file = os.path.join(self.args.output_dir, "probe_indices.json")
        self.probe_cache_file = probe_cache_file

        self.probe_indices = []
        self.probe_ready = False

        self.add_callback(ProbeRetainLambdaCallback(self, dual_warmup_epochs))

    def _load_probe_indices(self):
        if not self.probe_cache_file or not os.path.exists(self.probe_cache_file):
            return None
        try:
            with open(self.probe_cache_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        if isinstance(payload, dict):
            cached_method = str(payload.get("probe_selection_method", "random")).lower()
            if cached_method != self.probe_selection_method:
                return None
            payload = payload.get("probe_indices")
        if not isinstance(payload, list):
            return None
        n = len(self.full_retain_dataset)
        indices = [int(i) for i in payload if 0 <= int(i) < n]
        return indices if len(indices) > 0 else None

    def _save_probe_indices(self):
        if not self.probe_cache_file or not self.is_world_process_zero():
            return
        os.makedirs(os.path.dirname(self.probe_cache_file), exist_ok=True)
        payload = {
            "probe_indices": [int(i) for i in self.probe_indices],
            "probe_budget_k": int(self.probe_budget_k),
            "probe_selection_method": self.probe_selection_method,
        }
        with open(self.probe_cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @torch.no_grad()
    def _build_random_probe_indices(self):
        n = len(self.full_retain_dataset)
        if n == 0:
            return []
        k = max(1, min(self.probe_budget_k, n))
        generator = torch.Generator()
        generator.manual_seed(self.probe_seed)
        perm = torch.randperm(n, generator=generator)[:k]
        return perm.tolist()

    def _select_farthest_unused(self, embeddings, anchor_indices, used_mask):
        n = embeddings.size(0)
        if n == 0:
            return None
        if not anchor_indices:
            candidates = torch.where(~used_mask)[0]
            if candidates.numel() == 0:
                return None
            return int(candidates[0].item())
        anchors = embeddings[torch.tensor(anchor_indices, dtype=torch.long)]
        sims = torch.matmul(embeddings, anchors.T)
        nearest_sim = sims.max(dim=1).values
        masked = nearest_sim.clone()
        masked[used_mask] = float("inf")
        idx = int(torch.argmin(masked).item())
        if used_mask[idx]:
            return None
        return idx

    def _farthest_first_probe(self, embeddings, k):
        n = embeddings.size(0)
        if n == 0:
            return []
        k = int(max(1, min(k, n)))
        mean_vec = F.normalize(embeddings.mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0)
        first_idx = int(torch.argmin(1.0 - torch.matmul(embeddings, mean_vec)).item())
        selected = [first_idx]
        if k == 1:
            return selected
        selected_mask = torch.zeros(n, dtype=torch.bool)
        selected_mask[first_idx] = True
        min_dist = 1.0 - torch.matmul(embeddings, embeddings[first_idx])
        for _ in range(1, k):
            masked_dist = min_dist.clone()
            masked_dist[selected_mask] = -1.0
            next_idx = int(torch.argmax(masked_dist).item())
            selected.append(next_idx)
            selected_mask[next_idx] = True
            dist_to_new = 1.0 - torch.matmul(embeddings, embeddings[next_idx])
            min_dist = torch.minimum(min_dist, dist_to_new)
        return selected

    def _kmedoids_probe(self, embeddings, k, max_iter):
        n = embeddings.size(0)
        if n == 0:
            return []
        k = int(max(1, min(k, n)))
        medoid_indices = self._farthest_first_probe(embeddings=embeddings, k=k)
        for _ in range(max_iter):
            medoid_tensor = torch.tensor(medoid_indices, dtype=torch.long)
            medoid_emb = embeddings[medoid_tensor]
            sims = torch.matmul(embeddings, medoid_emb.T)
            assignments = torch.argmax(sims, dim=1)
            used_mask = torch.zeros(n, dtype=torch.bool)
            new_medoids = []
            for c in range(k):
                members = torch.where(assignments == c)[0]
                chosen_idx = None
                if members.numel() > 0:
                    sub = embeddings[members]
                    dist_matrix = 1.0 - torch.matmul(sub, sub.T)
                    cost = dist_matrix.sum(dim=1)
                    rank = torch.argsort(cost)
                    ordered_members = members[rank]
                    for idx in ordered_members.tolist():
                        if not used_mask[idx]:
                            chosen_idx = int(idx)
                            break
                if chosen_idx is None:
                    chosen_idx = self._select_farthest_unused(
                        embeddings, new_medoids, used_mask
                    )
                if chosen_idx is None:
                    continue
                used_mask[chosen_idx] = True
                new_medoids.append(chosen_idx)
            if len(new_medoids) < k:
                remain = torch.where(~used_mask)[0].tolist()
                new_medoids.extend([int(i) for i in remain[: (k - len(new_medoids))]])
            if new_medoids == medoid_indices:
                break
            medoid_indices = new_medoids
        return medoid_indices

    @torch.no_grad()
    def _build_retain_embeddings(self):
        self.model.eval()
        retain_loader = DataLoader(
            self.full_retain_dataset,
            batch_size=self.probe_build_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )
        all_repr = []
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )
        for batch in retain_loader:
            batch = self._prepare_inputs(batch)
            with autocast_ctx:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                    output_hidden_states=True,
                )
            hidden_states = outputs.hidden_states[-1]
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = F.normalize(pooled.float(), p=2, dim=-1)
            all_repr.append(pooled.cpu())
            del outputs, hidden_states, mask, pooled
        self.model.train()
        if len(all_repr) == 0:
            return None
        return torch.cat(all_repr, dim=0)

    @torch.no_grad()
    def _build_probe_indices(self):
        if self.probe_selection_method == "random":
            return self._build_random_probe_indices()
        embeddings = self._build_retain_embeddings()
        if embeddings is None or embeddings.size(0) == 0:
            return self._build_random_probe_indices()
        if self.probe_selection_method == "kmedoids":
            return self._kmedoids_probe(
                embeddings=embeddings,
                k=self.probe_budget_k,
                max_iter=self.probe_cluster_max_iter,
            )
        raise ValueError(
            f"Unsupported probe_selection_method: {self.probe_selection_method}. "
            "Use one of: random, kmedoids."
        )

    def ensure_probe_set_ready(self):
        if self.probe_ready and len(self.probe_indices) > 0:
            return
        cached = self._load_probe_indices()
        if cached is not None:
            self.probe_indices = cached[: self.probe_budget_k]
            self.probe_ready = len(self.probe_indices) > 0
            return
        self.probe_indices = self._build_probe_indices()
        self.probe_ready = len(self.probe_indices) > 0
        if self.probe_ready:
            self._save_probe_indices()
        logger.info("Probe set ready: size=%d", len(self.probe_indices))

    @torch.no_grad()
    def evaluate_probe_retain_loss(self):
        self.ensure_probe_set_ready()
        if not self.probe_indices:
            return None

        prev_mode = self.model.training
        self.model.eval()
        probe_dataset = Subset(self.full_retain_dataset, self.probe_indices)
        probe_loader = DataLoader(
            probe_dataset,
            batch_size=self.probe_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )

        losses = []
        for batch in probe_loader:
            batch = self._prepare_inputs(batch)
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
            }
            per_sample_nll, _ = compute_batch_nll(self.model, inputs)
            valid_tokens = (inputs["labels"][..., 1:] != -100).sum(dim=-1).clamp(min=1)
            losses.append((per_sample_nll / valid_tokens.float()).cpu())

        if prev_mode:
            self.model.train()

        if len(losses) == 0:
            return None
        return torch.cat(losses, dim=0).mean().item()

    def update_retain_lambda_with_probe(self, update_lambda=True):
        probe_loss = self.evaluate_probe_retain_loss()
        if probe_loss is None:
            logger.warning("Probe loss is unavailable, skip lambda update.")
            return

        shifted = probe_loss - self.retain_loss_eps
        prev_alpha = float(self.alpha)
        if update_lambda:
            self.alpha = max(0.0, self.alpha + self.dual_step_size * shifted)

        self.log(
            {
                "probe_retain_loss": probe_loss,
                "probe_size": len(self.probe_indices),
                "retain_lambda_shift": shifted,
                "retain_lambda_prev": prev_alpha,
                "retain_lambda": float(self.alpha),
            }
        )
        logger.info(
            "Probe lambda update: probe_loss=%.4f, alpha %.4f -> %.4f",
            probe_loss,
            prev_alpha,
            float(self.alpha),
        )


class ProbeRetainNPO(ProbeRetainLambdaMixin, NPO):
    """NPO + probe-loss-based dynamic retain lambda."""

    pass


class ProbeRetainSimNPO(ProbeRetainLambdaMixin, SimNPO):
    """SimNPO + probe-loss-based dynamic retain lambda."""

    pass


class ProbeRetainLambdaCallback(TrainerCallback):
    def __init__(self, trainer, dual_warmup_epochs=0):
        self.trainer = trainer
        self.dual_warmup_epochs = int(dual_warmup_epochs)

    def on_train_begin(self, args, state, control, **kwargs):
        update_lambda = self.dual_warmup_epochs == 0
        self.trainer.update_retain_lambda_with_probe(update_lambda=update_lambda)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0
        if epoch % self.trainer.probe_eval_interval != 0:
            return
        update_lambda = (state.epoch or 0.0) >= self.dual_warmup_epochs
        self.trainer.update_retain_lambda_with_probe(update_lambda=update_lambda)
