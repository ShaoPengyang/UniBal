<h1 align="center">
<b>UniBal</b>: A Unified Balancing Framework for Gradient-based LLM Unlearning
</h1>

<div align="center">
  <a href="https://github.com/nxZhai/BalDRO"><img src="https://img.shields.io/badge/GitHub-BalDRO_(prior_work)-94c320?logo=github"></a> &nbsp;
  <a href='https://arxiv.org/pdf/2601.09172'><img src='https://img.shields.io/badge/arXiv-2601.09172_(BalDRO)-red?logo=arXiv'></a> &nbsp;
</div>

<br>

> **UniBal** is the extended journal version of our conference paper **[BalDRO](https://github.com/nxZhai/BalDRO)** (WWW 2026). UniBal is currently under review.

## Introduction

Gradient-based LLM unlearning aims to remove target knowledge from trained LLMs by directly updating model parameters. However, existing methods often suffer from two imbalance issues:

- **Outer-level forget-retain imbalance**: the optimization needs of forgetting and retention evolve unevenly during training, but existing methods combine them with a fixed trade-off coefficient.
- **Inner-level forget-sample imbalance**: forget samples exhibit heterogeneous unlearning difficulty — some are easily forgotten while others require sustained optimization.

**UniBal** addresses both issues through a unified balancing framework:

| Level | Mechanism | Description |
| :---: | :--- | :--- |
| **Outer** | Semantic prototype-guided primal-dual optimization | Formulates retention as an explicit utility constraint and introduces a dual variable that adaptively adjusts the forget-retain trade-off. A compact probe retain set constructed via k-medoids provides stable retain-side feedback for dual updates. |
| **Inner** | KL-regularized distributionally robust optimization | Replaces the standard averaged forget loss with a Donsker-Varadhan log-sum-exp objective, automatically assigning larger weights to hard-to-forget samples. |

UniBal is **agnostic to the base unlearning objective** and can be instantiated on top of methods such as NPO, SimNPO, and SatImp. Experiments on **TOFU** and **MUSE** show that UniBal consistently improves both forget quality and model utility. For example, on TOFU (forget ratio = 1%), UniBal improves the forget quality of SimNPO from 0.4046 to 0.7659 (+89.3%) while also improving model utility from 0.5643 to 0.5966.

## Setup

### Install Dependencies

```bash
conda create -n unibal python=3.11.13
conda activate unibal

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Configure Weights & Biases

Training logs are reported via [Weights & Biases](https://wandb.ai/). After installing dependencies, log in with your account:

```bash
wandb login
```

### Dataset Preparation

TOFU and MUSE benchmarks are used for evaluation.

- [TOFU](https://huggingface.co/datasets/locuslab/TOFU)
- [MUSE-News](https://huggingface.co/datasets/muse-bench/MUSE-News) / [MUSE-Books](https://huggingface.co/datasets/muse-bench/MUSE-Books)

```bash
huggingface-cli download --repo-type dataset locuslab/TOFU
huggingface-cli download --repo-type dataset muse-bench/MUSE-News
huggingface-cli download --repo-type dataset muse-bench/MUSE-Books
```

### Prepare the Original Models

We use the official original models provided by [Open-Unlearning](https://github.com/locuslab/open-unlearning) and MUSE.

```bash
huggingface-cli download open-unlearning/tofu_Llama-2-7b-chat-hf_full
huggingface-cli download muse-bench/MUSE-books_target
huggingface-cli download muse-bench/MUSE-news_target
```

## Unlearning

### UniBal (Ours)

UniBal can be applied on top of different base unlearning objectives. The following scripts contain the configurations used in our experiments:

```bash
# ---- TOFU ----
# NPO + UniBal
bash scripts/unlearn/tofu/train_tofu_uni_npo.sh

# SimNPO + UniBal
bash scripts/unlearn/tofu/train_tofu_uni_simnpo.sh

# GroupNPO + UniBal
bash scripts/unlearn/tofu/train_tofu_uni_groupnpo.sh

# ---- MUSE ----
# NPO + UniBal
bash scripts/unlearn/muse/train_muse_uni_npo.sh

# SimNPO + UniBal
bash scripts/unlearn/muse/train_muse_uni_simnpo.sh

# SatImp + UniBal
bash scripts/unlearn/muse/train_muse_uni_satimp.sh
```

### Baseline Methods

#### NPO / SimNPO

```bash
# NPO
bash scripts/unlearn/tofu/train_tofu_npo.sh

# SimNPO
bash scripts/unlearn/tofu/train_tofu_simnpo.sh
```

#### BalDRO (Prior Work)

[BalDRO](https://github.com/nxZhai/BalDRO) is our prior conference work that focuses on inner-level sample-wise balancing via distributionally robust optimization. It includes two variants: **BalDRO-G** (GroupDRO-based) and **BalDRO-DV** (Donsker-Varadhan dual).

```bash
# ---- BalDRO-G (GroupDRO) ----
# NPO + BalDRO-G
bash scripts/unlearn/tofu/train_tofu_groupnpo.sh

# SimNPO + BalDRO-G
bash scripts/unlearn/tofu/train_tofu_groupsimnpo.sh

# SatImp + BalDRO-G
bash scripts/unlearn/tofu/train_tofu_groupsatimp.sh

# ---- BalDRO-DV (Donsker-Varadhan dual) ----
# NPO + BalDRO-DV
bash scripts/unlearn/tofu/train_tofu_drnpo.sh

# SimNPO + BalDRO-DV
bash scripts/unlearn/tofu/train_tofu_drsimnpo.sh

# SatImp + BalDRO-DV
bash scripts/unlearn/tofu/train_tofu_drsatimp.sh
```

## Evaluation

Evaluation is performed at every epoch during training. By default, we report **Forget Quality (FQ)** and **Model Utility (MU)**. You can customize evaluation metrics in `configs/muse.yaml` and `configs/tofu.yaml`.

For the performance of the **Original Model** and **Retain Model**, we evaluate using the retain model provided by Open-Unlearning. Results can be found in the `saves/eval` directory.

## Acknowledgements

This work builds upon [Open-Unlearning](https://github.com/locuslab/open-unlearning). We thank the authors for their contributions to the research community.

<!-- ## Citation

If you find our work useful, please consider citing both UniBal and BalDRO:

```bibtex
@article{shao2026unibal,
  title={UniBal: A Unified Balancing Framework for Gradient-based LLM Unlearning},
  author={Shao, Pengyang and Jin, Yanzheng and Shen, Fei and Kawaguchi, Kenji and Yang, Xun and Wang, Meng},
  year={2026}
}

@inproceedings{shao2026baldro,
  title={BalDRO: A Distributionally Robust Optimization based Framework for Large Language Model Unlearning},
  author={Shao, Pengyang and Zhai, Naixin and Chen, Lei and Yang, Yonghui and Zhu, Fengbin and Yang, Xun and Wang, Meng},
  booktitle={Proceedings of the ACM Web Conference 2026},
  year={2026}
}
``` -->

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
