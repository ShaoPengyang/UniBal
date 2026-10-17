<h1 align="center">UniBal</h1>

<p align="center"><strong>A Unified Balancing Framework for Gradient-based LLM Unlearning</strong></p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#datasets-and-models">Datasets</a> ·
  <a href="#train-on-tofu-and-muse">Training</a> ·
  <a href="#parameter-sweeps">Parameter sweeps</a> ·
  <a href="#evaluation-and-outputs">Evaluation</a>
</p>

UniBal is an LLM unlearning implementation built on the BalDRO codebase. It combines a distributionally robust forget objective with a constrained retain objective. Its **semantic prototype probe** summarizes retain-set behavior for the epoch-level dual update, using each prototype's cluster size as its weight.

> **Method at a glance:** robust forget loss → model updates with randomly sampled retain examples → cluster-weighted retain probe → projected update of the retain penalty.

The robust forget objective comes from BalDRO-DV, and the constrained primal–dual formulation follows PDU. UniBal's new component is the semantic prototype estimator for retain-side feedback. The prototypes are **only used to update the dual variable**; ordinary retain mini-batches still provide gradients for model training.

## Quick start

### 1. Install the environment

The supplied configurations target **Linux, NVIDIA CUDA, BF16, and FlashAttention 2**. They use Python 3.11 and pin the main Python dependencies in [`requirements.txt`](requirements.txt). Make sure your GPU and CUDA setup support these settings.

```bash
cd /path/to/UniBal
conda create -n unibal python=3.11.13 -y
conda activate unibal
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install flash-attn --no-build-isolation
```

If your machine requires a particular CUDA-enabled PyTorch wheel, install the compatible PyTorch 2.4.1 build before installing the rest of the requirements. Authenticate with Hugging Face and obtain access to the Llama checkpoints used below:

```bash
huggingface-cli login
python -c "import torch, transformers, hydra; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

The Python sweep launcher can print commands on Windows, but the pinned training stack and supplied shell wrappers are intended for Linux GPU use.

### 2. Inspect a run

Run commands from the repository root. `--dry-run` prints the complete Hydra command without loading a model or dataset:

```bash
python scripts/unlearn/sweep_unibal.py tofu npo --split forget01 --max-runs 1 --dry-run
```

### 3. Start training

```bash
python scripts/unlearn/sweep_unibal.py tofu npo --split forget01 --max-runs 1 --cuda-devices 0
```

`--max-runs 1` selects the first configuration in the method's default search profile. Remove it to execute the whole profile. For an initial one-epoch training check, add `--epochs 1 --no-eval --max-runs 1`; `--no-eval` skips benchmark evaluation but leaves the prototype-based dual update active.

## How UniBal works

For a forget mini-batch, the inherited BalDRO-DV objective aggregates per-example losses with a positive log-sum-exp term:

```text
balanced forget loss = β × log(mean(exp(forget_loss_i / β)))
```

The model is trained with this forget loss and a retain NLL term weighted by the current nonnegative dual variable `λ`. Random retain examples are sampled for these model updates. Separately, a frozen encoder embeds the retain examples; k-medoids selects real examples as semantic prototypes. At the end of each epoch, UniBal evaluates the selected prototypes and updates `λ`:

```text
probe retain loss = Σ_m (cluster_size_m / retain_set_size) × NLL(prototype_m)
λ_next           = max(0, λ + η × (probe retain loss − ε))
```

The probe uses **cluster proportions**, rather than a plain average over prototypes. `ε` is the retain-loss threshold and `η` is the dual step size. The default prototype count is 40 and the initial `λ` is 1. Implementations are in [`src/trainer/unlearn/unibal.py`](src/trainer/unlearn/unibal.py) and [`src/trainer/unlearn/prototypes.py`](src/trainer/unlearn/prototypes.py).

The implementation limits k-medoids optimization to at most 5,000 candidate retain examples by default, then assigns **all** retain examples to the chosen medoids to compute cluster weights. This bounds the clustering cost but is an approximation to full-set k-medoids. Change the candidate cap with `--medoid-sample-size`; change the requested prototype count with `--prototypes`.

## Datasets and models

Datasets are loaded through Hugging Face `datasets` and downloaded on first use. The launcher chooses the dataset, target checkpoint, tokenizer, and bundled retain-reference log from `--split`:

| Setting | Forget / retain training split | Starting checkpoint | Model config |
| --- | --- | --- | --- |
| TOFU `forget01` | `forget01` / `retain99` | `open-unlearning/tofu_Llama-2-7b-chat-hf_full` | `Llama-2-7b-chat-hf` |
| TOFU `forget05` | `forget05` / `retain95` | Same TOFU checkpoint | `Llama-2-7b-chat-hf` |
| TOFU `forget10` | `forget10` / `retain90` | Same TOFU checkpoint | `Llama-2-7b-chat-hf` |
| MUSE `News` | `forget` / `retain1` from `muse-bench/MUSE-News` | `muse-bench/MUSE-news_target` | `Llama-2-7b-hf` |
| MUSE `Books` | `forget` / `retain1` from `muse-bench/MUSE-Books` | `muse-bench/MUSE-books_target` | `ICLM-7B` |

MUSE News uses a Llama-2-7B backbone; MUSE Books uses ICLM-7B. The MUSE target checkpoints use the `meta-llama/Llama-2-7b-hf` tokenizer, supplied separately by the launcher. The MUSE `retain2` split is used for evaluation rather than retain-side training.

To stage datasets before running on a compute node:

```bash
huggingface-cli download --repo-type dataset locuslab/TOFU
huggingface-cli download --repo-type dataset muse-bench/MUSE-News
huggingface-cli download --repo-type dataset muse-bench/MUSE-Books
```

## Train on TOFU and MUSE

Choose `npo`, `simnpo`, or `satimp` as the forget objective. All three can be used with the TOFU and MUSE splits shown above. These examples each launch **one** configuration:

```bash
# TOFU: choose the forgetting proportion explicitly.
python scripts/unlearn/sweep_unibal.py tofu npo --split forget01 --max-runs 1 --cuda-devices 0
python scripts/unlearn/sweep_unibal.py tofu simnpo --split forget05 --max-runs 1 --cuda-devices 0
python scripts/unlearn/sweep_unibal.py tofu satimp --split forget10 --max-runs 1 --cuda-devices 0

# MUSE: select the News or Books target and its matching data.
python scripts/unlearn/sweep_unibal.py muse npo --split News --max-runs 1 --cuda-devices 0
python scripts/unlearn/sweep_unibal.py muse simnpo --split Books --max-runs 1 --cuda-devices 0
python scripts/unlearn/sweep_unibal.py muse satimp --split News --max-runs 1 --cuda-devices 0
```

For example, `muse satimp --split Books` or `tofu npo --split forget10` are also valid. The shell wrappers under `scripts/unlearn/tofu/` and `scripts/unlearn/muse/` call the same launcher and accept its options.

## Parameter sweeps

The launcher runs the Cartesian product of the values you provide. Default profiles are compact starting grids adapted from earlier experiment scripts; **they are not the manuscript's full search space**. Use `--dry-run` to review the combinations, `--max-runs N` to cap them, and `--continue-on-error` to keep going after a failed run.

```bash
# Preview the first three combinations of a paper-style TOFU NPO grid.
python scripts/unlearn/sweep_unibal.py tofu npo --split forget01 \
  --lr 1e-5 2e-5 3e-5 4e-5 5e-5 \
  --batch 8 --accum 2 4 \
  --beta-dv 1 2 5 10 --eps 0.12 0.14 0.16 0.18 \
  --eta 0.02 0.05 0.1 0.2 --max-runs 3 --dry-run

# Run a smaller Books sweep; remove --max-runs to execute every combination.
python scripts/unlearn/sweep_unibal.py muse simnpo --split Books \
  --lr 2e-5 5e-5 --beta-dv 1 2 --eps 0.14 0.16 \
  --eta 0.05 0.1 --max-runs 4 --cuda-devices 0
```

For a single-GPU run, the effective batch size is `--batch × --accum`. With distributed training, multiply by the number of training processes. `--cuda-devices` only sets visible devices; it does not start a distributed launcher. The manuscript's shared search space includes learning rates `1e-5` through `5e-5`, effective batch sizes 16 and 32, DV temperatures 1/2/5/10, `ε` values 0.12/0.14/0.16/0.18, and `η` values 0.02/0.05/0.1/0.2. The examples above do not sweep every base-method parameter; add `--beta` for NPO/SimNPO, `--delta` for SimNPO, or `--beta1` and `--beta2` for SatImp as needed.

| Option | Controls | Default |
| --- | --- | --- |
| `--prototypes` | Number of semantic retain representatives | 40 |
| `--medoid-sample-size` | Candidate count for k-medoids optimization | 5,000 |
| `--probe-batch` | Batch size for the epoch-end probe | TOFU 10; MUSE 2 |
| `--encoder` | Frozen semantic encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| `--encoder-batch-size`, `--encoder-max-length` | Embedding batch size and token limit | 32; 256 |
| `--eps`, `--eta`, `--beta-dv` | Retain threshold, dual step, and robust forget temperature | Method profile |
| `--lr`, `--batch`, `--accum`, `--epochs` | Optimizer and training schedule | Method profile |
| `--model-path`, `--model-config`, `--tokenizer-path` | Override target checkpoint/config/tokenizer | Dataset defaults |
| `--retain-logs-path` | Matching retain-only evaluation log | Bundled reference path |
| `--output-root`, `--report-to` | Run directory and logging integration | `saves/unlearn`; `none` |

`--beta-dv` controls the robust aggregation and is distinct from `--beta`, which belongs to NPO/SimNPO. The forget-term multiplier `--gamma` defaults to 1.0. The launcher requires `--retain-logs-path` when `--model-path` is changed and evaluation is enabled, to avoid silently comparing against the wrong reference.

## Evaluation and outputs

Each configuration has its own directory under:

```text
saves/unlearn/<benchmark>/<split>/<trainer>/<configuration>/
```

The run directory contains the saved model, Trainer state, and `retain_prototypes.json` cache. Unless `--no-eval` is set, the default evaluation suite runs at each epoch and after training; metrics are written to checkpoint `evals/` directories as `TOFU_EVAL.json` / `TOFU_SUMMARY.json` or `MUSE_EVAL.json` / `MUSE_SUMMARY.json`.

Default TOFU evaluation includes model utility, forget quality, and gibberish metrics. Default MUSE evaluation includes knowledge memorization, verbatim memorization, privacy leakage, and extraction strength. The metric lists are configured in [`configs/eval/tofu.yaml`](configs/eval/tofu.yaml) and [`configs/eval/muse.yaml`](configs/eval/muse.yaml). **The default suites do not enable every metric reported in the manuscript.** Enable the additional entries needed for a full table before running those evaluations.

Some metrics compare against a retain-only reference evaluation log. The launcher selects a bundled log for each default split; a custom target model needs a matching log supplied with `--retain-logs-path`. The bundled MUSE Books log uses a legacy `Llama-2-7b-hf` directory name. Confirm the checkpoint provenance of any reference log before using it for paper-level comparisons. The custom benchmark evaluator runs on one accelerator process; for multi-process training, run evaluation separately on one process.

## Code map and provenance

| Component | Location |
| --- | --- |
| UniBal primal loss and epoch-end dual update | [`src/trainer/unlearn/unibal.py`](src/trainer/unlearn/unibal.py) |
| Semantic encoding, k-medoids, and cluster weights | [`src/trainer/unlearn/prototypes.py`](src/trainer/unlearn/prototypes.py) |
| NPO, SimNPO, SatImp with BalDRO-DV aggregation | `src/trainer/unlearn/{npo,simnpo,satimp}.py` |
| Random forget/retain training pairs | [`src/data/unlearn.py`](src/data/unlearn.py) |
| Trainer defaults | `configs/trainer/UniBal*.yaml` |
| Sweep launcher | [`scripts/unlearn/sweep_unibal.py`](scripts/unlearn/sweep_unibal.py) |
| Source-level paper/code audit | [`docs/UNIBAL_PAPER_AUDIT.md`](docs/UNIBAL_PAPER_AUDIT.md) |

The original BalDRO baseline scripts remain under `scripts/unlearn/tofu/` and `scripts/unlearn/muse/`. UniBal inherits their BalDRO-DV forget aggregation and uses their codebase as its foundation; its semantic prototype feedback is implemented in the UniBal modules above. This repository also builds on [Open-Unlearning](https://github.com/locuslab/open-unlearning), [BalDRO](https://github.com/nxZhai/BalDRO), and the PDU formulation cited in the manuscript.

This checkout has been checked at source level and with launcher dry runs. Full 7B training, benchmark evaluation, and numerical reproduction still require a compatible GPU environment and verified reference checkpoints. See the [paper/code audit](docs/UNIBAL_PAPER_AUDIT.md) for the precise scope and known approximations.

## Citation and license

The UniBal manuscript has no public citation link yet. For work using the inherited BalDRO framework, cite its [paper](https://arxiv.org/abs/2601.09172):

```bibtex
@inproceedings{shao2026baldro,
  title={Baldro: A distributionally robust optimization based framework for large language model unlearning},
  author={Shao, Pengyang and Zhai, Naixin and Chen, Lei and Yang, Yonghui and Zhu, Fengbin and Yang, Xun and Wang, Meng},
  booktitle={Proceedings of the ACM Web Conference 2026},
  pages={8874--8884},
  year={2026}
}
```

See [`LICENSE`](LICENSE) for the license.
