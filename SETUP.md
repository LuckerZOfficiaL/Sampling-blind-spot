# Setup

This guide walks through installing the code and downloading the models /
datasets used in the paper.

## 1. Prerequisites

- Python 3.11
- A CUDA-capable GPU (3B–12B models require ~12–48 GB VRAM at bf16)
- A Hugging Face account and access token
  (https://huggingface.co/settings/tokens). Llama and Mistral models gate
  their weights behind a one-click access form.

## 2. Clone and install

```bash
git clone <repo-url> sampling-blind-spot
cd sampling-blind-spot

python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## 3. Set environment variables

The codebase resolves model and dataset paths via two env vars:

```bash
export MODELS_DIR=/where/you/want/to/store/checkpoints
export DATASETS_DIR=/where/you/want/to/store/datasets
export HF_TOKEN=hf_...                  # your Hugging Face token
```

Persist them in your shell init file or in a `.env` you `source` before
each session.

## 4. Download models

```bash
for model in \
    Qwen/Qwen2.5-3B-Instruct \
    meta-llama/Llama-3.2-3B-Instruct \
    meta-llama/Llama-3.1-8B-Instruct \
    mistralai/Mistral-Nemo-Instruct-2407
do
    huggingface-cli download "$model" \
        --local-dir "$MODELS_DIR/$(basename "$model")" \
        --cache-dir "$MODELS_DIR"
done
```

## 5. Download datasets

```bash
python <<'PY'
import os
from datasets import load_dataset

root = os.environ["DATASETS_DIR"]

# GSM8K
load_dataset("openai/gsm8k", "main").save_to_disk(f"{root}/gsm8k")

# MATH
load_dataset("lighteval/MATH", "all").save_to_disk(f"{root}/math")

# MMLU-Pro
load_dataset("TIGER-Lab/MMLU-Pro", split="test").save_to_disk(f"{root}/mmlu_pro")
PY
```

## 6. Smoke test

Verify the grafting hook installs and a baseline pass completes:

```bash
python experiments/validate_hooks.py --model "$MODELS_DIR/Qwen2.5-3B-Instruct"

python experiments/run_paper_replication.py \
    --experiments baselines \
    --model-b "$MODELS_DIR/Qwen2.5-3B-Instruct" \
    --baseline-models b \
    --benchmark gsm8k --n-samples 10 --seed 42 \
    --output-dir results/smoke_test
```

## 7. (Optional) Configure W&B logging

```bash
cp .env.local .env       # add WANDB_API_KEY=...
```

Pass `--wandb` to experiment scripts to enable logging.

## Next steps

See [README.md](README.md) for the full reproduction recipe (sampling
baselines, deterministic regime sweep, $R_k$ recovery table, mechanism
analysis).
