# Sampling Blind Spot in Math-Reasoning Difficulty Estimation

Code for the paper **"Hard or Just Unreached? Diagnosing the Sampling
Blind Spot in Math-Reasoning Difficulty Estimation."**

This project uses activation grafting as a **diagnostic** for
constructing cheap, deterministic alternatives to sampling on a single
model, then uses that deterministic regime to probe what items
`pass@k=0` actually labels as hard. The grafting infrastructure builds
on the [Communicating Activations Between Language Model
Agents](https://arxiv.org/abs/2501.14082) codebase.

> **TL;DR.** Across four open-weight instruction-tuned models on GSM8K
> and MATH, 10.3–22.9% of items no sampling seed solves in six tries are
> reached at matched compute by a six-chain deterministic regime
> (greedy plus five cheap residual-stream perturbations). Pass@k=0 is
> therefore decoding-regime-dependent, not an intrinsic property of the
> items.

## Overview

`pass@k=0` (no sampled chain reaches gold in $k$ tries) is the canonical
hardness signal for math/reasoning evaluation: it drives RL-with-verifiable-rewards
filters, math/code data curation, difficulty-stratified curricula, and
verifier-training datasets.

We show this signal has a **persistent blind spot**. On the eight
free-form math cells we test (GSM8K and MATH across four open-weight
instruction-tuned models — `Qwen-2.5-3B`, `Llama-3.2-3B`,
`Llama-3.1-8B`, `Mistral-Nemo-12B`), of items that no sampling seed
in $\{42,\dots,47\}$ solves, **$10.3$–$22.9\%$ are reached at matched
compute by a six-chain deterministic regime** (greedy plus five cheap
residual-stream perturbations applied via activation grafting), while
greedy alone solves at most $6\%$ on these cells.

Activation grafting is used purely as a diagnostic tool to diversify the
deterministic regime — not as something to deploy at inference.

## The deterministic regime

The deterministic regime consists of greedy chains that differ only in a
single last-prompt-token activation replacement at layer $\ell=26$, fired
during prefill only. All graft vectors are fixed across the dataset (no
per-example optimisation, parameter-free).

| Symbol | Description |
|---|---|
| $g_B$ | Greedy baseline (no graft) |
| $g_Z$ / `zero` | Zero vector |
| $g_R$ / `rand` | Norm-matched Gaussian |
| `runit` | Random unit direction |
| `shuf` | Hidden-dim permutation of the baseline activation |
| `bos` | BOS-token activation |
| `avg` | Mean over prompt-token activations |
| `prev` | Same position from layer $\ell-1$ |

These are mechanistically distinct (cross-kind fix-set Jaccard $\leq 0.47$
in every (model, benchmark) cell).

## Models & benchmarks

| Model | $d_{\mathrm{model}}$ | Layers |
|---|---|---|
| `Qwen-2.5-3B-Instruct` | 2048 | 36 |
| `Llama-3.2-3B-Instruct` | 3072 | 28 |
| `Llama-3.1-8B-Instruct` | 4096 | 32 |
| `Mistral-Nemo-12B-Instruct` | 5120 | 40 |

| Benchmark | Task | Scoring |
|---|---|---|
| GSM8K | Math word problems | Numeric match |
| MATH | Competition math | `math_equal()` symbolic equivalence |
| MMLU-Pro | Multiple-choice reasoning (A–J) | Exact letter match |

All cells use $n=1000$ matched prompts, `max_new_tokens=2048`.

## Installation

See [SETUP.md](SETUP.md) for the step-by-step install and
model/dataset download guide. Short version:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
export MODELS_DIR=/path/to/checkpoints
export DATASETS_DIR=/path/to/datasets
```

All scripts read `MODELS_DIR` and `DATASETS_DIR` from the environment;
substitute Hugging Face hub names (e.g. `Qwen/Qwen2.5-3B-Instruct`) if
you prefer the loader to pull from the hub on first use.

## Reproducing the results

The main evaluation entry point is `experiments/run_paper_replication.py`.

### 1. Sampling baseline (six seeds, $T=0.7$, $p=0.9$)

```bash
for seed in 42 43 44 45 46 47; do
    python experiments/run_paper_replication.py \
        --experiments baselines --baseline-models a \
        --model-a $MODELS_DIR/Qwen2.5-3B-Instruct \
        --model-b $MODELS_DIR/Qwen2.5-3B-Instruct \
        --benchmark gsm8k --n-samples 1000 --seed $seed \
        --output-dir results_sampling/seed_${seed}_qwen3B_gsm8k
done
```

(Repeat across the four models × three benchmarks; see
`scripts/run_scaleup_phase2_sampling.sh`.)

### 2. Deterministic regime (greedy + five grafts)

The "AC" experiment path runs greedy with a residual-stream graft hook;
the graft vector is selected via `--combination-fn`. Repeat for each graft
kind:

```bash
for graft in baseline zero random_act random_unit shuffled_act bos_token average_act prev_layer; do
    python experiments/run_paper_replication.py \
        --experiments ac \
        --model-b $MODELS_DIR/Qwen2.5-3B-Instruct \
        --combination-fn $graft --graft-position last \
        --layer-a 26 --layer-b 26 \
        --benchmark gsm8k --n-samples 1000 --seed 42 \
        --output-dir results_greedy_decoding/qwen3B_gsm8k_${graft}_L26
done
```

`scripts/run_scaleup_phase1_grafts.sh` runs the eight grafts in batch.

### 3. Compute $R_k$ recovery on the pass@$6=0$ slice

```bash
python experiments/_compute_k6_matched.py
```

This walks the `results_*/` directories, intersects sampling-fail with
deterministic-success per item, and emits Table 3 of the paper.

### 4. Mechanism: hidden-state divergence + attention deltas

```bash
python experiments/analyze_attention.py \
    --model $MODELS_DIR/Qwen2.5-3B-Instruct \
    --benchmark gsm8k --n-samples 100 \
    --combination-fns random_act zero
```

Produces the per-layer $L_2$ / cosine trajectory in Figure 4.

### 5. Fix-set Jaccard (per-cell + per-pair)

```bash
python experiments/_diversity_jaccard.py
```

Emits the cross-kind ceiling ($\leq 0.47$) and same-axis ceiling tables.

### 6. Label-free probe (PR curves, lift)

```bash
python experiments/_h1_disagreement_routing.py
```

## Project layout

```
sampling-blind-spot/
├── src/
│   ├── communication/
│   │   ├── activation_graft.py        # Core grafting engine (hook)
│   │   └── combination_functions.py   # Per-kind graft vector construction
│   ├── models/
│   │   ├── activation_extractor.py    # Hook-based extraction
│   │   ├── model_loader.py            # HF loading + device map
│   │   └── model_registry.py          # Per-model d_model / num_layers
│   ├── benchmarks/
│   │   ├── reasoning_tasks.py         # GSM8K, MATH, MMLU-Pro loaders
│   │   └── mmlu_handler.py            # MMLU-Pro MCQ formatting
│   └── evaluation/
│       └── metrics.py                 # Numerical match, math_equal
├── experiments/
│   ├── run_paper_replication.py       # Main entry point (sampling + grafting)
│   ├── analyze_attention.py           # Eager-attn instrumentation
│   ├── layer_grid_search.py           # Layer ablation sweep
│   ├── validate_hooks.py              # Hook installation sanity check
│   ├── _compute_k6_matched.py         # R_k recovery table
│   ├── _diversity_jaccard.py          # Fix-set Jaccard
│   ├── _h1_disagreement_routing.py    # Label-free probe lift / AUPRC
│   ├── fig4_hidden_divergence.py      # Figure: L2/cosine divergence
│   └── fig5_rk_payoff.py              # Figure: R_k recovery bars
├── scripts/
│   ├── run_scaleup_phase1_grafts.sh   # SLURM template: 8 graft kinds
│   └── run_scaleup_phase2_sampling.sh # SLURM template: 6 sampling seeds
└── tests/
```

## Configuration

Experiments are configured via CLI flags; see
`experiments/run_paper_replication.py --help`. Key flags:

| Flag | Description |
|---|---|
| `--combination-fn {baseline,zero,random_act,random_unit,shuffled_act,bos_token,average_act,prev_layer}` | Which graft kind to inject |
| `--graft-position {first,last}` | Which prompt token to graft at |
| `--layer-a / --layer-b` | Extraction / injection layer (default 26 for the main results) |
| `--benchmark {gsm8k,math,mmlu_pro,...}` | Reasoning benchmark |
| `--n-samples` | Number of prompts |
| `--seed` | Sampling seed (set $T>0$ for stochastic decoding) |

Decoding mode: `--temperature 0.0` is greedy (used for all deterministic
chains); $T=0.7$ + $p_{\mathrm{top}}=0.9$ for sampling.

## Tests

```bash
pytest tests/
```

## Acknowledgments

This codebase builds on the cross-model activation-communication
infrastructure of:

```bibtex
@article{communicating-activations-2025,
  title   = {Communicating Activations Between Language Model Agents},
  journal = {arXiv preprint arXiv:2501.14082},
  year    = {2025}
}
```
