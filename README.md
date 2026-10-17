# BalDRO · UniBal

<p align="center">
  <strong>Distributionally robust LLM unlearning with semantic prototype feedback</strong><br>
  <a href="https://arxiv.org/abs/2601.09172">BalDRO paper</a> ·
  <a href="https://www.overleaf.com/project/6992dd33bd6f8274e561efa9">UniBal manuscript</a> ·
  <a href="https://github.com/nxZhai/BalDRO">Upstream BalDRO</a>
</p>

This repository contains the original BalDRO training framework and a **UniBal implementation rebuilt on its BalDRO-DV trainers**. UniBal combines a robust forget loss with a constrained retain objective. A small set of semantic retain prototypes supplies feedback for the dual variable after each epoch.

![BalDRO framework](assets/baldro_overview.png)

## Contents

- [Installation](#installation)
- [Data and checkpoints](#data-and-checkpoints)
- [Train UniBal](#train-unibal)
- [Run parameter sweeps](#run-parameter-sweeps)
- [How the implementation maps to the paper](#how-the-implementation-maps-to-the-paper)
- [Outputs and evaluation](#outputs-and-evaluation)
- [Original BalDRO scripts](#original-baldro-scripts)
- [Reproducibility notes](#reproducibility-notes)

## Installation

Use **Linux with an NVIDIA GPU** for the supplied training configuration. The project pins PyTorch 2.4.1, Transformers 4.45.1, DeepSpeed 0.15.4, and other versions in [`requirements.txt`](requirements.txt). The model configs use BF16 and FlashAttention 2; choose a CUDA driver and GPU that support both. Multi-GPU runs need enough aggregate memory for the 7B target model, the reference model used by NPO, optimizer state, and evaluation.

```bash
cd BalDRO-main  # this modified checkout
conda create -n baldro python=3.11.13 -y
conda activate baldro
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install flash-attn --no-build-isolation
```

Install a CUDA-compatible PyTorch build before the remaining requirements if your platform needs a specific wheel. Log in to Hugging Face and accept the relevant Llama model license before training:

```bash
huggingface-cli login
python -c "import torch, transformers, hydra; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

The sweep launcher is Python and also works as a command generator on Windows. The pinned GPU training stack and shell wrappers are intended for Linux.

## Data and checkpoints

Hugging Face datasets are downloaded through `datasets` on first use. Download them in advance if your cluster has no network access:

```bash
huggingface-cli download --repo-type dataset locuslab/TOFU
huggingface-cli download --repo-type dataset muse-bench/MUSE-News
huggingface-cli download --repo-type dataset muse-bench/MUSE-Books
```

| Benchmark | Forget / retain training data | Starting checkpoint | Tokenizer | Bundled retain reference log |
| --- | --- | --- | --- | --- |
| TOFU `forget01` | `forget01` / `retain99` | `open-unlearning/tofu_Llama-2-7b-chat-hf_full` | Same checkpoint | `saves/eval/tofu_Llama-2-7b-chat-hf_retain99/TOFU_EVAL.json` |
| TOFU `forget05` | `forget05` / `retain95` | Same | Same | `saves/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json` |
| TOFU `forget10` | `forget10` / `retain90` | Same | Same | `saves/eval/tofu_Llama-2-7b-chat-hf_retain90/TOFU_EVAL.json` |
| MUSE News | `forget` / `retain1` | `muse-bench/MUSE-news_target` (Llama-2-7B) | `meta-llama/Llama-2-7b-hf` | `saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json` |
| MUSE Books | `forget` / `retain1` | `muse-bench/MUSE-books_target` (ICLM-7B) | `meta-llama/Llama-2-7b-hf` | `saves/eval/muse_Llama-2-7b-hf_Books_retrain/MUSE_EVAL.json` |

The MUSE target checkpoints do not package tokenizer files; the launcher supplies the benchmark's Llama-2 tokenizer separately. The Books reference log has a **legacy `Llama-2-7b-hf` directory name** in this repository. Its filename is not evidence of checkpoint provenance. Check that a reference log was produced from the corresponding retain-only model before using its comparison metrics in a paper; pass a verified log with `--retain-logs-path` when needed. MUSE `retain2` is an evaluation split; the training retain split is `retain1`.

## Train UniBal

Run from the repository root. The launcher translates each option into a Hydra training command and gives each run a separate output directory. `--max-runs 1` starts one configuration from the profile. The commands below cover all three forget objectives and all three primary dataset settings:

```bash
# TOFU: select forget01, forget05, or forget10.
python scripts/unlearn/sweep_unibal.py tofu npo    --split forget01 --max-runs 1 --cuda-devices 0
python scripts/unlearn/sweep_unibal.py tofu simnpo --split forget05 --max-runs 1 --cuda-devices 0
python scripts/unlearn/sweep_unibal.py tofu satimp --split forget10 --max-runs 1 --cuda-devices 0

# MUSE: select News or Books. Each checkpoint uses its paper-specified backbone.
python scripts/unlearn/sweep_unibal.py muse npo    --split News  --max-runs 1 --cuda-devices 0
python scripts/unlearn/sweep_unibal.py muse simnpo --split Books --max-runs 1 --cuda-devices 0
python scripts/unlearn/sweep_unibal.py muse satimp --split News  --max-runs 1 --cuda-devices 0
```

Use `--dry-run` first to inspect the exact command without loading models or downloading data:

```bash
python scripts/unlearn/sweep_unibal.py muse npo --split Books --max-runs 1 --dry-run
```

For a short smoke run, add `--epochs 1 --max-runs 1 --no-eval`. The full evaluation suite is time consuming. The optional shell wrappers in `scripts/unlearn/tofu/` and `scripts/unlearn/muse/` forward their arguments to the same Python launcher, for example `bash scripts/unlearn/tofu/train_tofu_unibal_npo.sh --split forget01 --max-runs 1`.

### Main training options

| Option | Meaning | Default |
| --- | --- | --- |
| `--prototypes` | Requested number of retain clusters / representative samples | `40`, as in the manuscript |
| `--probe-batch` | Number of prototypes evaluated at a time | TOFU `10`, MUSE `2` |
| `--encoder` | Frozen semantic encoder checkpoint | `sentence-transformers/all-MiniLM-L6-v2` |
| `--medoid-sample-size` | Maximum number of candidate points used to optimize medoids | `5000` |
| `--encoder-batch-size`, `--encoder-max-length` | Semantic encoding batch size and token limit | `32`, `256` |
| `--eps` | Retain-loss constraint threshold, ε | Dataset and method profile |
| `--eta` | Epoch-level dual step size, η | Dataset and method profile |
| `--beta-dv` | BalDRO-DV forget temperature, β | Dataset and method profile |
| `--lr`, `--batch`, `--accum`, `--epochs` | Optimizer learning rate, per-device batch, gradient accumulation, epochs | Dataset and method profile |
| `--model-path`, `--model-config`, `--tokenizer-path` | Override checkpoint, Hydra architecture config, or tokenizer | Benchmark defaults above |
| `--retain-logs-path` | Verified retain-only evaluation JSON for the selected model/split | Bundled path above |
| `--report-to` | Trainer logging integration (`none`, `wandb`, etc.) | `none` |

For NPO and SimNPO, `--beta` is the **base forget-loss scale** and differs from `--beta-dv`. SimNPO also accepts `--delta`; SatImp accepts `--beta1` and `--beta2`. `--gamma` scales the forget term and defaults to `1.0` to match the manuscript objective.

## Run parameter sweeps

Each argument with multiple values forms a Cartesian product. The built-in profiles are **practical starting grids based on the old UniBal scripts**, with corrected BalDRO-DV and prototype weighting; they are not the entire paper search space. Supply explicit ranges to reproduce the manuscript's shared grid:

```bash
# Preview a paper-style TOFU NPO grid; batch 8 × accumulation 2/4
# gives effective batch sizes 16/32 on one GPU.
python scripts/unlearn/sweep_unibal.py tofu npo --split forget01 \
  --lr 1e-5 2e-5 3e-5 4e-5 5e-5 \
  --batch 8 --accum 2 4 \
  --beta-dv 1 2 5 10 --eps 0.12 0.14 0.16 0.18 \
  --eta 0.02 0.05 0.1 0.2 --max-runs 3 --dry-run

# Start a smaller MUSE Books sweep and keep going if one run fails.
python scripts/unlearn/sweep_unibal.py muse simnpo --split Books \
  --lr 2e-5 5e-5 --beta-dv 1 2 --eps 0.14 0.16 \
  --eta 0.05 0.1 --cuda-devices 0 --continue-on-error
```

An effective batch size is `--batch × --accum × number_of_GPUs`. Use `--max-runs N` to cap a search; remove it for the complete Cartesian product. The launcher runs combinations sequentially and exits on the first failed run unless `--continue-on-error` is set. Runs are **independent**: each starts from the original target checkpoint.

## How the implementation maps to the paper

| Paper step | Implementation | Detail |
| --- | --- | --- |
| Forget-side KL-DRO / DV loss | `src/trainer/unlearn/{npo,simnpo,satimp}.py` | Positive `β log mean exp(lossᵢ / β)` emphasizes higher-loss forget examples. |
| Stochastic retain gradient | `src/data/unlearn.py`, `src/trainer/unlearn/unibal.py` | Model updates use randomly selected retain examples and mean per-example next-token NLL. |
| Semantic prototypes | `src/trainer/unlearn/prototypes.py` | Frozen encoder embeds retain question–answer pairs or MUSE text chunks; k-medoids chooses real examples. |
| Cluster-weighted probe | `src/trainer/unlearn/prototypes.py`, `src/trainer/unlearn/unibal.py` | Each medoid's NLL is weighted by `cluster_size / retain_set_size`; it is **not** a plain average of medoids. |
| Projected dual update | `src/trainer/unlearn/unibal.py` | After each epoch: `λ ← max(0, λ + η × (weighted_probe_loss − ε))`, starting at `λ = 1`. |
| Primal loss | `src/trainer/unlearn/{npo,simnpo,satimp}.py` | Robust forget loss plus `λ × retain_loss`; the constant `−λε` has no model gradient. |

The requested prototype count is `min(40, number_of_nonempty_retain_examples)` by default; empty clusters can reduce the final count. The `--prototypes` option changes it directly. To control the quadratic k-medoids cost, the implementation optimizes medoids over at most **5,000 candidate retain examples**, then assigns **all** retain examples to the selected medoids before calculating weights. This candidate sampling is a computational approximation to the manuscript's full-set k-medoids statement. Change the cap with `--medoid-sample-size`. A cached `retain_prototypes.json` stores indices, weights, encoder identity, dataset hash, and settings for reuse.

The old UniBal scripts were used only as a source of useful hyperparameter ranges. The old forget weighting and unweighted mean over representative points are not used by this implementation. Method configs are in `configs/trainer/UniBal*.yaml`; the sweep orchestration is in `scripts/unlearn/sweep_unibal.py`.

## Outputs and evaluation

The default output root is `saves/unlearn/<benchmark>/<split>/<trainer>/<configuration>/`. Each run writes a checkpoint and Trainer state; the prototype cache is in the same run directory. Evaluation runs at every epoch and once after training, unless `--no-eval` is set. The default suites are configured in `configs/eval/tofu.yaml` and `configs/eval/muse.yaml` and write `TOFU_EVAL.json` / `TOFU_SUMMARY.json` or `MUSE_EVAL.json` / `MUSE_SUMMARY.json` under the run directory.

TOFU defaults to model utility, forget quality, and gibberish metrics. MUSE defaults to knowledge and verbatim memorization, privacy leakage, and extraction strength. Comparison metrics use a **retain-only reference evaluation log**; its model and split must match the run. Passing a different checkpoint requires a corresponding `--retain-logs-path`. For the manuscript's full metric tables, enable the additional metrics listed in `configs/eval/tofu.yaml` and `configs/eval/muse.yaml` before running evaluation. The default evaluation config alone does not produce every metric named in the paper. The custom evaluation code runs on a single accelerator process; multi-process training requires a separate single-process evaluation run.

## Original BalDRO scripts

The original NPO, SimNPO, and SatImp scripts remain under `scripts/unlearn/{tofu,muse}/`. `train_*_drnpo.sh`, `train_*_drsimnpo.sh`, and `train_*_drsatimp.sh` run BalDRO-DV; `train_*_group*.sh` run the group variant. Those scripts encode the original BalDRO experiment settings and are separate from the UniBal sweep launcher.

## Reproducibility notes

- The source-level audit is in [`docs/UNIBAL_PAPER_AUDIT.md`](docs/UNIBAL_PAPER_AUDIT.md). It distinguishes direct formula matches, approximations, and items requiring a GPU run.
- Training the 7B checkpoints and the full evaluation suite has not been validated in this checkout. Start with `--dry-run`, then a one-epoch run on your compute node.
- For TOFU and MUSE references, verify the exact retain-only checkpoint behind each bundled JSON before reporting paper-level comparisons. The Books log directory uses a legacy model label.

## Acknowledgements and citation

This code builds on [BalDRO](https://github.com/nxZhai/BalDRO) and [Open-Unlearning](https://github.com/locuslab/open-unlearning). If you use the original BalDRO framework, cite:

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
