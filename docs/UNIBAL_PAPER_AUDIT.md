# UniBal paper-to-code audit

Audit scope: the UniBal method in the supplied manuscript (`sec/03model.tex`, `sec/04experiments.tex`) against the trainers, datasets, configurations, sweep launcher, and default evaluation in this checkout. This is a source-level audit; it is not a claim that the manuscript's numerical results have been reproduced.

## Method correspondence

| Manuscript requirement | Code location | Finding |
| --- | --- | --- |
| `min_θ L_f^bal + λ(L_r − ε)` with `λ ≥ 0` | `src/trainer/unlearn/{npo,simnpo,satimp}.py`, `src/trainer/unlearn/unibal.py` | **Matches for model gradients.** The constant `−λε` is omitted from the computed primal loss because it has zero derivative with respect to θ. `alpha` is set to the current dual variable. `gamma` defaults to 1.0. |
| DV forget aggregation `β log mean_i exp(ℓ_i/β)` | `DrNPO`, `DrSimNPO`, `DrSatImp` | **Matches.** The sign emphasizes the larger per-example forget losses. The original BalDRO-main aggregation is retained. |
| Random retain mini-batch for each primal model update | `src/data/unlearn.py` | **Matches.** `ForgetRetainDataset` samples a retain item when a forget item is read. |
| Mean per-example next-token retain NLL | `src/trainer/unlearn/unibal.py::compute_retain_loss` | **Matches.** Sequence NLL is divided by the number of valid shifted target tokens, then averaged across examples. |
| Frozen semantic encoder; real-example k-medoids representatives | `src/trainer/unlearn/prototypes.py` | **Matches in construction**, subject to the candidate sampling approximation below. Question and answer are embedded together for TOFU; MUSE uses its tokenized text chunks. |
| Probe weight `|C_m| / |D_r|` | `k_medoids_prototypes`, `update_dual` | **Matches.** All nonempty retain examples are assigned to selected medoids, and each prototype NLL receives its cluster proportion. A plain mean over medoids would be incorrect. |
| Epoch-level projected dual update, initialized at one | `_DualUpdateCallback`, `update_dual` | **Matches.** One update at each epoch end: `max(0, λ + η(probe_loss − ε))`; λ starts at 1. |
| `M = 40` | `configs/trainer/UniBal*.yaml`, sweep launcher | **Matches by default.** Fewer than 40 nonempty retain examples yield fewer prototypes. |
| TOFU / MUSE dataset and backbone protocol | `scripts/unlearn/sweep_unibal.py`, `configs/model/ICLM-7B.yaml` | **Matches configured model choices.** TOFU uses Llama-2-7B Chat; News uses Llama-2-7B; Books uses MUSE's ICLM-7B target checkpoint. |

## Resolved discrepancies in this audit

1. **MUSE checkpoint names:** The launcher now uses the actual lowercase `MUSE-news_target` and `MUSE-books_target` repositories while retaining the uppercase dataset names (`MUSE-News`, `MUSE-Books`).
2. **Books backbone:** Added a distinct `ICLM-7B` model config and selected it for MUSE Books. The MUSE target checkpoint itself is still loaded from Hugging Face; the config provides the tokenizer/template and model-loading settings.
3. **SimNPO objective scale:** UniBalSimNPO and its sweep profiles now use `gamma = 1.0`, the unscaled forget term written in the manuscript. `--gamma` remains available for explicitly labeled ablations.
4. **MUSE text chunking:** An exact multiple of `max_length` previously produced an empty last chunk. The chunk count now uses ceiling division, so prototype indices and retain sample counts refer to real chunks.
5. **Books reference log path:** The launcher points to the existing bundled Books retrain JSON, whose directory has a legacy `Llama-2-7b-hf` label. That filename does not establish that the retrained checkpoint used the ICLM-7B backbone.

## Explicit approximations and verification limits

- **K-medoids candidate cap.** The manuscript describes clustering the retain set. The code selects medoids from at most 5,000 candidate examples to bound quadratic distance work, then assigns the complete retain set to those medoids. Thus the probe weights reflect the complete retain set, but the medoids may differ from full-set k-medoids. The cap is `medoid_sample_size` in the UniBal trainer configs.
- **Paper grid versus launch defaults.** The manuscript searches learning rates `1e-5` to `5e-5`, effective batch sizes 16/32, DV temperatures 1/2/5/10, ε 0.12/0.14/0.16/0.18, and η 0.02/0.05/0.1/0.2. The launcher defaults are smaller starting grids inherited from old experiment scripts; pass these ranges explicitly for a paper-style sweep. Base-method parameters are additional axes.
- **Reference log provenance.** The included retain-only JSON files have not been tied to model checkpoint hashes. In particular, the bundled Books retrain log uses a legacy directory name. A paper-grade comparison needs a verified log from the matching retain-only model and split, passed with `--retain-logs-path` if different.
- **Evaluation coverage.** The default TOFU and MUSE suites enable subsets of the manuscript's reported metrics. Additional metrics must be enabled in `configs/eval/tofu.yaml` and `configs/eval/muse.yaml`. The custom evaluator in `src/trainer/base.py` executes only on one accelerator process; distributed training does not currently yield the same evaluation path.
- **Runtime verification.** Static inspection and launcher dry runs do not validate 7B checkpoint loading, CUDA/FlashAttention compatibility, DeepSpeed behavior, or numerical reproduction. These require a GPU environment with the pinned dependencies and access to the checkpoints.

## Scope boundary

The original BalDRO and other baseline trainers remain present as comparison code. This audit checked the paths used by `UniBalNPO`, `UniBalSimNPO`, `UniBalSatImp`, their data loaders, the sweep launcher, and their default evaluation. It does not certify every unrelated baseline trainer or every metric implementation in the upstream framework.
