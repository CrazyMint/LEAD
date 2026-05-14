<div align="center">

# LEAD: Length-Efficient Adaptive and Dynamic Reasoning for Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2605.09806-b31b1b.svg?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.09806)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg?style=flat&logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/pdf/2605.09806)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![vLLM](https://img.shields.io/badge/vLLM-rollouts-30A14E.svg?style=flat)](https://github.com/vllm-project/vllm)
[![HuggingFace](https://img.shields.io/badge/🤗_Model-Kotom1%2Fmath__lead__4k-yellow.svg?style=flat)](https://huggingface.co/Kotom1/math_lead_4k_deepseek-r1-1.5b)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://github.com/CrazyMint/LEAD/pulls)

</div>

Reference implementation for the paper **"LEAD: Length-Efficient Adaptive and Dynamic Reasoning for Large Language Models"** ([arXiv:2605.09806](https://arxiv.org/abs/2605.09806)).

LEAD is a self-calibrating multi-reward RL framework for **efficient reasoning**, built on top of [verl](https://github.com/volcengine/verl). It replaces the two static knobs of GRPO-style efficient-reasoning recipes — fixed reward weights and a global length budget — with two online, self-calibrating mechanisms:

1. **🎛️ Dynamic reward weighting with decoupled group normalization** — each reward is normalized in its own rollout group before combination, and the combination weights are updated online via a Potential-Scaled Instability (PSI) controller. No hand-tuned schedule.
2. **📏 Per-problem online target-length calibration** — the global length budget is replaced by a per-prompt target $L^*_q$ estimated from the model's own correct rollouts, with a symmetric efficiency reward around $L^*_q$ that penalizes both overthinking and over-compression.

LEAD is a drop-in replacement for the GRPO advantage; just set `algorithm.adv_estimator=lead` in the verl config.

---

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🛠️ Requirements](#%EF%B8%8F-requirements)
- [📦 Installation](#-installation)
- [⚙️ Configure Environment](#%EF%B8%8F-configure-environment)
- [📚 Data Preparation](#-data-preparation)
- [🏋️ Training](#%EF%B8%8F-training)
- [📈 Evaluation](#-evaluation)
- [🤗 Released Checkpoints](#-released-checkpoints)
- [📁 Repository Layout](#-repository-layout)
- [🎚️ Key Configuration Knobs](#%EF%B8%8F-key-configuration-knobs)
- [📊 Reproducing Paper Tables](#-reproducing-paper-tables)
- [🧰 Troubleshooting](#-troubleshooting)
- [📝 Citation](#-citation)
- [📄 License](#-license)

---

## 🚀 Quick Start

```bash
# 1. Install
conda create -n lead python=3.10 -y && conda activate lead
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install flash-attn --no-build-isolation
pip install "vllm>=0.6.0"

# 2. Configure (HF token for DeepSeek-R1-Distill download)
cp .env.example .env
$EDITOR .env

# 3. Train LEAD on DeepSeek-R1-Distill-Qwen-1.5B (4K budget)
bash train_math_lead_deepseek-r1-1.5b.sh
```

---

## 🛠️ Requirements

| Component | Tested version |
|---|---|
| OS | Ubuntu 22.04 |
| Python | 3.10 |
| CUDA | 12.1 |
| PyTorch | 2.4+ |
| GPU | 4× NVIDIA L40S (44 GB) or A6000 (48 GB) for 1.5B; 8× A6000 for 7B |
| RAM | 256 GB recommended |
| Disk | NVMe SSD; ~150 GB for model weights + 50 GB per checkpoint |

---

## 📦 Installation

```bash
# Clone
git clone https://github.com/CrazyMint/LEAD.git LEAD-release
cd LEAD-release

# Create env
conda create -n lead python=3.10 -y
conda activate lead

# PyTorch with CUDA 12.1
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# verl + LEAD (this repo)
pip install -e .

# Required runtime extras (not pulled by setup.py)
pip install flash-attn --no-build-isolation
pip install "vllm>=0.6.0"
```

If `pip install -e .` fails on `requirements.txt`, install dependencies directly: `pip install -r requirements.txt`.

---

## ⚙️ Configure Environment

The training scripts run `source "$(dirname $0)/.env"` on startup, so this file is required. Create it from the template:

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```bash
export WANDB_API_KEY="..."   # optional; logs go to wandb
export HF_TOKEN="..."        # required to download DeepSeek-R1-Distill weights
export HF_HOME="/path/to/hf-cache"   # optional (defaults to ~/.cache/huggingface)
```

Sanity-check the HF token works:

```bash
huggingface-cli whoami   # should print your username
```

---

## 📚 Data Preparation

The training scripts read `data/math/{train,test}.parquet`. **These files are checked into the repo**, so you can skip this step on first run.

To regenerate from scratch:

```bash
bash scripts/data/download_math.sh
# or directly:
python scripts/data/prepare_math.py --local_dir data/math
```

The shipped files are the MATH level 3–5 split (8,521 problems) used in the paper.

---

## 🏋️ Training

### LEAD on DeepSeek-R1-Distill-Qwen-1.5B (Table 1, 4K budget)

```bash
bash train_math_lead_deepseek-r1-1.5b.sh
```

On first run this downloads the base model (~7 GB) into `$HF_HOME`, launches Ray + vLLM, and trains for 7 epochs (462 steps). On 4× L40S 44 GB the run takes ~30 wall-clock hours. Checkpoints are written to `${OUTPUT_ROOT:-./results}/math_lead_4k_deepseek-r1-1.5b/`.

To run on a different GPU count, override before invocation:

```bash
N_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash train_math_lead_deepseek-r1-1.5b.sh
```

> ⚠️ **Note:** the paper used 4 GPUs; results may shift slightly with smaller world size due to per-batch group statistics.

### GRPO baseline (for the Table 1 comparison)

```bash
bash train_math_grpo_deepseek-r1-1.5b.sh
```

---

## 📈 Evaluation

The paper uses [Sober Reasoning](https://github.com/sober-reasoning/sober) settings (temperature 0.8, top-p 0.9, pass@n where n=3 for MATH-500/Olympiad and n=10 for AIME 24/25 and AMC 23). Any verl- or lighteval-compatible eval harness will work. This release does not yet ship a turn-key evaluation script.

---

## 🤗 Released Checkpoints

| Paper row | HuggingFace repo |
|---|---|
| Table 1, LEAD 1.5B-4K (Acc 53.36 / Len 3714 / AES 0.68) | [`Kotom1/math_lead_4k_deepseek-r1-1.5b`](https://huggingface.co/Kotom1/math_lead_4k_deepseek-r1-1.5b) |

Quick load:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("Kotom1/math_lead_4k_deepseek-r1-1.5b")
t = AutoTokenizer.from_pretrained("Kotom1/math_lead_4k_deepseek-r1-1.5b")
```

---

## 📁 Repository Layout

```
LEAD-release/
├── verl/                                # patched verl (adv_estimator='lead')
│   ├── trainer/ppo/ray_trainer.py       # LEAD branch (Algorithm 1 in the paper)
│   ├── trainer/config/ppo_trainer.yaml
│   └── utils/reward_score/deepscale.py
├── train_math_lead_deepseek-r1-1.5b.sh  # paper Table 1 (LEAD)
├── train_math_grpo_deepseek-r1-1.5b.sh  # paper Table 1 (GRPO baseline)
├── ablations/
│   ├── math_grpo_lambda_sweep/          # paper Table 2 (GRPO column)
│   ├── math_lead_lambda_sweep/          # paper Table 2 (LEAD-static column)
│   └── math_lead_aggregator_ablation/   # paper Table 3
├── data/
│   ├── math/                            # MATH (shipped)
│   ├── deepscaler/                      # alt training set (optional)
│   └── ...
└── scripts/
    └── data/                            # download + prepare scripts
```

---

## 🎚️ Key Configuration Knobs

In `verl/trainer/config/ppo_trainer.yaml` (override on the command line via `algorithm.<key>=<value>`):

| Flag | Default | Description |
|---|---|---|
| `algorithm.adv_estimator` | `gae` | Set to `lead` to enable LEAD |
| `algorithm.lead_alpha` | `1.0` | Potential-decay exponent ($\alpha$) |
| `algorithm.lead_beta` | `0.95` | EMA momentum for weight smoothing |
| `algorithm.lead_lambda_min` | `0.3` | Floor on $\lambda_c$ after EMA |
| `algorithm.lead_bmax` | `8000` | Sentinel for unsolved prompts; matches training-time max length |
| `algorithm.lead_lstar_mode` | `max_asym` | `max_sym` (paper), `max_asym`, or `upper_only` |
| `algorithm.lead_aggregator` | `mean_correct` | `mean_correct` (paper), `min_correct`, `median_correct`, `mean_all` |
| `algorithm.lead_static_lambda_corr` | `null` | If set, bypass dynamic weights and use a fixed $(\lambda_c, \lambda_\ell)$ — used by the static-vs-dynamic ablation |

---

## 📊 Reproducing Paper Tables

| Table | Command |
|---|---|
| Table 1, LEAD 1.5B-4K | `bash train_math_lead_deepseek-r1-1.5b.sh` |
| Table 2, GRPO ratio sweep | `bash ablations/math_grpo_lambda_sweep/run_sweep.sh` |
| Table 2, LEAD-static ratio sweep | `METHODS=lead bash ablations/math_grpo_lambda_sweep/run_sweep.sh` |
| Table 3, $L^*_q$ aggregator | `bash ablations/math_lead_aggregator_ablation/run_sweep.sh` |

---

## 🧰 Troubleshooting

| Symptom | Fix |
|---|---|
| `source .env: No such file or directory` | Run the configure step (`cp .env.example .env`) and fill it in. |
| `HF gated repo` error when downloading DeepSeek-R1-Distill | Accept the model's terms on its HuggingFace page and ensure `HF_TOKEN` in `.env` is valid: `huggingface-cli whoami` should print your username. |
| `flash-attn` build fails | Ensure CUDA toolkit + `nvcc` are on `PATH` and you have at least 16 GB of build RAM. |
| Only one GPU available | Set `N_GPUS=1` and `CUDA_VISIBLE_DEVICES=0` before running. Results may shift slightly versus the 4-GPU paper setting. |
| vLLM OOM during rollout | Lower `actor_rollout_ref.rollout.gpu_memory_utilization` from `0.65` to `0.5` in the training script. |
| Ray complains about port already in use | Kill any stray Ray cluster: `ray stop --force` and re-run. |

---

## 📝 Citation

If LEAD is useful in your research, please cite our paper:

```bibtex
@misc{wei2026lead,
      title={LEAD: Length-Efficient Adaptive and Dynamic Reasoning for Large Language Models},
      author={Songtao Wei and Yi Li and Zhikai Li and Xu Hu and Yuede Ji and Guanpeng Li and Feng Chen and Carl Yang and Zhichun Guo and Bingzhe Li},
      year={2026},
      eprint={2605.09806},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.09806}
}
```

---

## 📄 License

Apache 2.0 (inherited from upstream verl). See [`LICENSE`](LICENSE).
