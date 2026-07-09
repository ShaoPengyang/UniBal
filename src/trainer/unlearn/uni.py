import json
import logging
import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import TrainerCallback

from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.npo import DrNPO, GroupNPO
from trainer.unlearn.simnpo import DrSimNPO
from trainer.unlearn.satimp import DrSatImp
from trainer.utils import compute_batch_nll

logger = logging.getLogger(__name__)


# ======================================================================
# Mixin – can be composed with any unlearning trainer that exposes
# self.alpha, self.train_dataset.forget / .retain, and self.data_collator
# ======================================================================
class UniMixin:
    """Adds full-retain evaluation, top-k hard mining, and primal-dual
    lambda adjustment to any unlearning trainer.

    At every epoch boundary the mixin:
    1. Evaluates the model on the **entire** retain set (per-sample NLL).
    2. Selects the *top_k_retain* samples with the highest average NLL.
    3. Updates the active retain subset so subsequent training batches
       draw only from these hard samples.
    4. Adjusts ``alpha`` (lambda) via a primal-dual rule::

           alpha <- max(0, alpha + dual_step_size * (avg_retain_loss - retain_loss_eps))

    ``compute_loss`` is **not** overridden – the parent class's loss
    function (including any DRO weighting) is used unchanged.  The
    dynamic ``alpha`` is picked up automatically.
    """

    def __init__(
        self,
        retain_loss_eps=0.0,
        dual_step_size=1.0,
        top_k_retain=None,
        dual_warmup_epochs=0,
        retain_eval_batch_size=16,
        retain_eval_ratio=1.0,
        retain_eval_interval=1,
        retain_scoring_mode="full_eval",
        retain_score_ema=0.2,
        retain_lambda_use_topk=False,
        log_retain_ids_txt=False,
        lambda_update_mode="legacy",
        probe_budget_k=None,
        probe_build_batch_size=None,
        probe_eval_batch_size=None,
        probe_cache_file=None,
        probe_selection_method="farthest_first",
        probe_cluster_max_iter=20,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.retain_loss_eps = retain_loss_eps
        self.dual_step_size = dual_step_size
        self.retain_eval_batch_size = retain_eval_batch_size
        self.retain_eval_ratio = retain_eval_ratio
        self.retain_eval_interval = max(1, int(retain_eval_interval))
        self.retain_scoring_mode = retain_scoring_mode
        self.retain_score_ema = retain_score_ema
        self.retain_lambda_use_topk = retain_lambda_use_topk
        self.lambda_update_mode = lambda_update_mode
        self.log_retain_ids_txt = log_retain_ids_txt
        self.retain_ids_txt_path = None
        if self.log_retain_ids_txt:
            self.retain_ids_txt_path = os.path.join(self.args.output_dir, "retain_ids_per_epoch.txt")
            os.makedirs(self.args.output_dir, exist_ok=True)
            if self.is_world_process_zero():
                with open(self.retain_ids_txt_path, "w", encoding="utf-8") as f:
                    f.write("")

        original_dataset = self.train_dataset
        forget_ds = original_dataset.forget
        retain_ds = original_dataset.retain
        self.full_retain_dataset = retain_ds
        self.retain_score_bank = torch.zeros(len(retain_ds), dtype=torch.float32)
        self.retain_score_seen = torch.zeros(len(retain_ds), dtype=torch.bool)
        self.retain_score_mean = 0.0

        if top_k_retain is None:
            top_k_retain = len(forget_ds)
        self.top_k_retain = top_k_retain
        if probe_budget_k is None:
            probe_budget_k = self.top_k_retain
        self.probe_budget_k = int(probe_budget_k)
        self.probe_build_batch_size = int(probe_build_batch_size or retain_eval_batch_size)
        self.probe_eval_batch_size = int(probe_eval_batch_size or retain_eval_batch_size)
        self.probe_cache_file = probe_cache_file or os.path.join(self.args.output_dir, "probe_indices.json")
        self.probe_selection_method = str(probe_selection_method).lower()
        self.probe_cluster_max_iter = max(1, int(probe_cluster_max_iter))
        self.probe_indices = []
        self.probe_ready = False

        # Keep the original unlearning dataset (ForgetRetainDataset) so retain
        # samples are drawn from the full retain set via random pairing.
        self.train_dataset = original_dataset

        self.add_callback(UniCallback(self, dual_warmup_epochs))

    def _log_retain_ids_to_txt(self, top_indices, k):
        if not self.log_retain_ids_txt:
            return
        if not self.is_world_process_zero():
            return
        if self.retain_ids_txt_path is None:
            return
        epoch = float(self.state.epoch) if getattr(self.state, "epoch", None) is not None else -1.0
        global_step = int(getattr(self.state, "global_step", -1))
        if isinstance(top_indices, torch.Tensor):
            top_indices = top_indices.detach().to("cpu").long().tolist()
        elif hasattr(top_indices, "tolist"):
            top_indices = top_indices.tolist()
        else:
            top_indices = list(top_indices)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "k": int(k),
            "retain_ids": top_indices,
        }
        with open(self.retain_ids_txt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def update_retain_score_bank(self, retain_indices, per_sample_losses):
        """Online update of retain sample scores from training batches."""
        if retain_indices is None or per_sample_losses is None:
            return
        idx_cpu = retain_indices.detach().to("cpu").long()
        loss_cpu = per_sample_losses.detach().to("cpu").float()
        for i, s in zip(idx_cpu.tolist(), loss_cpu.tolist()):
            if self.retain_score_seen[i]:
                old = self.retain_score_bank[i].item()
                self.retain_score_bank[i] = (1 - self.retain_score_ema) * old + self.retain_score_ema * s
            else:
                self.retain_score_bank[i] = s
                self.retain_score_seen[i] = True
        if loss_cpu.numel() > 0:
            batch_mean = loss_cpu.mean().item()
            self.retain_score_mean = (
                batch_mean if self.retain_score_mean == 0.0 else (1 - self.retain_score_ema) * self.retain_score_mean + self.retain_score_ema * batch_mean
            )

    def _load_probe_indices(self):
        if not self.probe_cache_file:
            return None
        if not os.path.exists(self.probe_cache_file):
            return None
        try:
            with open(self.probe_cache_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        if isinstance(payload, dict):
            cached_method = str(payload.get("probe_selection_method", "farthest_first")).lower()
            if cached_method != self.probe_selection_method:
                return None
            cached_budget = payload.get("probe_budget_k", None)
            if cached_budget is not None and int(cached_budget) != int(self.probe_budget_k):
                return None
            payload = payload.get("probe_indices")
        if not isinstance(payload, list):
            return None
        max_idx = len(self.full_retain_dataset) - 1
        indices = [int(i) for i in payload if 0 <= int(i) <= max_idx]
        if len(indices) == 0:
            return None
        return indices

    def _save_probe_indices(self):
        if not self.probe_cache_file:
            return
        if not self.is_world_process_zero():
            return
        os.makedirs(os.path.dirname(self.probe_cache_file), exist_ok=True)
        payload = {
            "probe_indices": [int(i) for i in self.probe_indices],
            "probe_selection_method": self.probe_selection_method,
            "probe_budget_k": int(self.probe_budget_k),
        }
        with open(self.probe_cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

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

    def _select_farthest_unused(self, embeddings, anchor_indices, used_mask):
        n = embeddings.size(0)
        if n == 0:
            return None
        if anchor_indices is None or len(anchor_indices) == 0:
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

    def _kmeans_probe(self, embeddings, k, max_iter):
        n = embeddings.size(0)
        if n == 0:
            return []
        k = int(max(1, min(k, n)))
        seed_indices = self._farthest_first_probe(embeddings=embeddings, k=k)
        centers = embeddings[torch.tensor(seed_indices, dtype=torch.long)].clone()
        assignments = None
        for _ in range(max_iter):
            sims = torch.matmul(embeddings, centers.T)
            new_assignments = torch.argmax(sims, dim=1)
            if assignments is not None and torch.equal(assignments, new_assignments):
                break
            assignments = new_assignments
            updated_centers = []
            for c in range(k):
                members = torch.where(assignments == c)[0]
                if members.numel() == 0:
                    fallback_idx = int(torch.argmin(sims.max(dim=1).values).item())
                    updated_centers.append(embeddings[fallback_idx])
                    continue
                centroid = embeddings[members].mean(dim=0, keepdim=True)
                centroid = F.normalize(centroid, p=2, dim=-1).squeeze(0)
                updated_centers.append(centroid)
            centers = torch.stack(updated_centers, dim=0)

        sims = torch.matmul(embeddings, centers.T)
        assignments = torch.argmax(sims, dim=1)
        selected = []
        used_mask = torch.zeros(n, dtype=torch.bool)
        for c in range(k):
            members = torch.where(assignments == c)[0]
            chosen_idx = None
            if members.numel() > 0:
                member_sims = torch.matmul(embeddings[members], centers[c])
                sorted_pos = torch.argsort(member_sims, descending=True)
                ordered_members = members[sorted_pos]
                for idx in ordered_members.tolist():
                    if not used_mask[idx]:
                        chosen_idx = int(idx)
                        break
            if chosen_idx is None:
                chosen_idx = self._select_farthest_unused(embeddings, selected, used_mask)
            if chosen_idx is None:
                continue
            used_mask[chosen_idx] = True
            selected.append(chosen_idx)
        if len(selected) < k:
            remain = torch.where(~used_mask)[0].tolist()
            selected.extend([int(i) for i in remain[: (k - len(selected))]])
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
                    chosen_idx = self._select_farthest_unused(embeddings, new_medoids, used_mask)
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
    def _build_probe_indices(self):
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.model.train()
        embeddings = torch.cat(all_repr, dim=0)
        if self.probe_selection_method == "farthest_first":
            return self._farthest_first_probe(embeddings=embeddings, k=self.probe_budget_k)
        if self.probe_selection_method == "kmeans":
            return self._kmeans_probe(
                embeddings=embeddings,
                k=self.probe_budget_k,
                max_iter=self.probe_cluster_max_iter,
            )
        if self.probe_selection_method == "kmedoids":
            return self._kmedoids_probe(
                embeddings=embeddings,
                k=self.probe_budget_k,
                max_iter=self.probe_cluster_max_iter,
            )
        raise ValueError(f"Unsupported probe_selection_method: {self.probe_selection_method}")

    def ensure_probe_set_ready(self):
        if self.lambda_update_mode != "probe":
            return
        if self.probe_ready and len(self.probe_indices) > 0:
            return
        cached = self._load_probe_indices()
        if cached is not None:
            self.probe_indices = cached
            self.probe_ready = True
            return
        self.probe_indices = self._build_probe_indices()
        self.probe_ready = len(self.probe_indices) > 0
        if self.probe_ready:
            self._save_probe_indices()
        logger.info("Probe set ready: size=%d", len(self.probe_indices))

    @torch.no_grad()
    def evaluate_probe_retain_loss(self):
        if len(self.probe_indices) == 0:
            return None
        self.model.eval()
        probe_dataset = Subset(self.full_retain_dataset, self.probe_indices)
        probe_loader = DataLoader(
            probe_dataset,
            batch_size=self.probe_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )
        losses = []
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )
        for batch in probe_loader:
            batch = self._prepare_inputs(batch)
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
                "use_cache": False,
            }
            with autocast_ctx:
                per_sample_nll, outputs = compute_batch_nll(self.model, inputs)
            valid_tokens = (inputs["labels"][..., 1:] != -100).sum(dim=-1).clamp(min=1)
            losses.append((per_sample_nll / valid_tokens.float()).cpu())
            del outputs, per_sample_nll
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.model.train()
        return torch.cat(losses, dim=0).mean().item()

    # ------------------------------------------------------------------
    # Full-retain evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_full_retain(self):
        """Return (losses, indices) where losses is per-sample average NLL.

        If retain_eval_ratio < 1, evaluate a random subset for speed/memory.
        `indices` are the dataset indices corresponding to returned losses.
        """
        # Defragment cache before full-retain pass to reduce OOM risk on 7B models.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.eval()

        total_n = len(self.full_retain_dataset)
        eval_n = max(1, int(total_n * float(self.retain_eval_ratio)))
        eval_n = min(eval_n, total_n)
        if eval_n < total_n:
            eval_indices = torch.randperm(total_n)[:eval_n]
            retain_dataset = Subset(self.full_retain_dataset, eval_indices.tolist())
        else:
            eval_indices = torch.arange(total_n)
            retain_dataset = self.full_retain_dataset

        retain_loader = DataLoader(
            retain_dataset,
            batch_size=self.retain_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )

        all_losses = []
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )
        for batch in retain_loader:
            batch = self._prepare_inputs(batch)
            inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
                # Disable KV cache during retain evaluation to lower peak memory.
                "use_cache": False,
            }
            with autocast_ctx:
                per_sample_nll, outputs = compute_batch_nll(self.model, inputs)
            valid_tokens = (
                (inputs["labels"][..., 1:] != -100).sum(dim=-1).clamp(min=1)
            )
            all_losses.append((per_sample_nll / valid_tokens.float()).cpu())
            # Explicitly release large tensors between mini-batches.
            del outputs, per_sample_nll
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.model.train()
        return torch.cat(all_losses, dim=0), eval_indices

    # ------------------------------------------------------------------
    # Top-k selection + lambda update
    # ------------------------------------------------------------------
    def update_retain_and_lambda(self, update_lambda=True):
        """Evaluate the full retain set, pick top-k hardest samples, and
        optionally adjust ``alpha`` (lambda) via the primal-dual rule."""
        self.ensure_probe_set_ready()
        if self.retain_scoring_mode == "online":
            seen = self.retain_score_seen
            if seen.any():
                scores = self.retain_score_bank.clone()
                unseen_fill = self.retain_score_mean if self.retain_score_mean > 0 else scores[seen].mean().item()
                scores[~seen] = unseen_fill
                k = min(self.top_k_retain, len(scores))
                top_k_values, top_indices = torch.topk(scores, k=k)
                avg_retain_loss = scores.mean().item()
                top_k_avg_loss = top_k_values.mean().item()
                retain_eval_samples = int(seen.sum().item())
            else:
                # Fallback before score bank is populated.
                all_indices = list(range(len(self.full_retain_dataset)))
                k = min(self.top_k_retain, len(all_indices))
                top_indices = all_indices[:k]
                avg_retain_loss = self.retain_score_mean
                top_k_avg_loss = self.retain_score_mean
                retain_eval_samples = 0
        else:
            per_sample_losses, eval_indices = self.evaluate_full_retain()
            k = min(self.top_k_retain, len(per_sample_losses))
            top_k_values, top_positions = torch.topk(per_sample_losses, k=k)
            top_indices = eval_indices[top_positions].tolist()
            avg_retain_loss = per_sample_losses.mean().item()
            top_k_avg_loss = top_k_values.mean().item()
            retain_eval_samples = len(per_sample_losses)

        prev_alpha = float(self.alpha)
        self._log_retain_ids_to_txt(top_indices, k)
        probe_loss = None
        if self.lambda_update_mode == "probe":
            probe_loss = self.evaluate_probe_retain_loss()
            if probe_loss is None:
                lambda_source_loss = avg_retain_loss
                lambda_source = "global"
            else:
                lambda_source_loss = probe_loss
                lambda_source = "probe"
        elif self.lambda_update_mode == "topk":
            lambda_source_loss = top_k_avg_loss
            lambda_source = "topk"
        elif self.lambda_update_mode == "global":
            lambda_source_loss = avg_retain_loss
            lambda_source = "global"
        else:
            use_topk = bool(getattr(self, "retain_lambda_use_topk", False))
            lambda_source_loss = top_k_avg_loss if use_topk else avg_retain_loss
            lambda_source = "topk" if use_topk else "global"
        shifted = lambda_source_loss - self.retain_loss_eps
        if update_lambda:
            self.alpha = max(0.0, self.alpha + self.dual_step_size * shifted)

        log_payload = {
            "retain_lambda": self.alpha, 
            "retain_avg_loss": avg_retain_loss,
            "top_k_avg_loss": top_k_avg_loss,
            "retain_lambda_source": lambda_source,
            "shift": shifted,
        }
        if probe_loss is not None:
            log_payload["probe_retain_loss"] = probe_loss
            log_payload["probe_size"] = len(self.probe_indices)
        self.log(log_payload)
        logger.info(
            "Retain update: full_loss=%.4f, top-%d_loss=%.4f, lambda=%.4f (src=%s)",
            avg_retain_loss, k, top_k_avg_loss, self.alpha,
            lambda_source,
        )


# ======================================================================
# Concrete classes – one-liners thanks to the mixin.
# compute_loss is inherited from the base Dr* / GradDiff method
# unchanged; the dynamic alpha is picked up automatically.
# ======================================================================

class UniPDU(UniMixin, GradDiff):
    """GradDiff + hard-retain mining + primal-dual lambda."""
    pass


class UniNPO(UniMixin, DrNPO):
    """DrNPO (BalDRO-NPO) + hard-retain mining + primal-dual lambda."""
    pass


class UniGroupNPO(UniMixin, GroupNPO):
    """GroupNPO + hard-retain mining + primal-dual lambda."""
    pass


class UniSimNPO(UniMixin, DrSimNPO):
    """DrSimNPO (BalDRO-SimNPO) + hard-retain mining + primal-dual lambda."""
    pass


class UniSatImp(UniMixin, DrSatImp):
    """DrSatImp (BalDRO-SatImp) + hard-retain mining + primal-dual lambda."""
    pass


# ======================================================================
# Callback that drives the epoch-boundary updates
# ======================================================================
class UniCallback(TrainerCallback):
    def __init__(self, trainer, dual_warmup_epochs=0):
        self.trainer = trainer
        self.dual_warmup_epochs = dual_warmup_epochs

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("Initial full-retain evaluation for top-k selection …")
        update_lambda = self.dual_warmup_epochs == 0
        self.trainer.update_retain_and_lambda(update_lambda=update_lambda)

    def on_epoch_end(self, args, state, control, **kwargs):
        # Optionally evaluate retain set every N epochs to reduce overhead.
        epoch_idx = int(state.epoch) if state.epoch is not None else 0
        if epoch_idx % self.trainer.retain_eval_interval != 0:
            return
        update_lambda = state.epoch >= self.dual_warmup_epochs
        self.trainer.update_retain_and_lambda(update_lambda=update_lambda)
