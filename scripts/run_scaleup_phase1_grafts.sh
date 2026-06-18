#!/bin/bash
#SBATCH --job-name=scaleup_p1
#SBATCH --time=12:00:00
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=logs/scaleup_p1_%A_%a.out
#SBATCH --error=logs/scaleup_p1_%A_%a.err
#SBATCH --array=0-75

# Template SLURM submission script. Edit:
#   * --account / --partition / --qos for your cluster
#   * env_setup.sh path (module load python, activate venv, etc.)
#   * MODELS_DIR env var (location of HuggingFace model checkpoints)
#   * ANCHOR_* paths below if you have precomputed greedy-baseline runs
#     for matched sample-ID enforcement; otherwise drop --match-samples-from

# Phase 1: deterministic graft chains for new (model, benchmark) cells.
# 10 cells x 7 grafts = 70 tasks. Greedy, layer 26, last-token graft.
# All cells anchor to existing predictions.json (no Phase-0 needed).
# Note: for cells with pre-existing gZ/gR (Qwen-3B mmlu_pro+medmcqa, Llama-8B mmlu_pro),
#   we re-run aligned to the new anchor for clean ID parity across the chain set.

# Source your environment-setup script (module load, venv activate, etc.):
# source /path/to/your/env_setup.sh
cd "$(dirname "$0")/.."  # repo root
# Benchmark-level anchors (IDs are model-agnostic).
ANCHOR_gsm8k="results_conf/gsm8k_baseline_Qwen2.5-3B-Instruct_greedy_1000/baseline_Qwen2_5_3B_Instruct/predictions.json"
ANCHOR_math="results_conf/math_baseline_Qwen2.5-3B-Instruct_greedy_1000/baseline_Qwen2_5_3B_Instruct/predictions.json"
ANCHOR_mmlu_pro="results_greedy_decoding/mmlu_pro_ac_zero_llama8B_1000/ac_trained_projection/predictions.json"
ANCHOR_medmcqa="results_no_truncation/medmcqa_ac_random_qwen3b_1000/ac_trained_projection/predictions.json"

# Cell list: model_tag, bench, model_path_suffix, model_name
CELLS=(
    "qwen mmlu_pro Qwen2.5-3B-Instruct      Qwen2.5-3B-Instruct"
    "qwen medmcqa  Qwen2.5-3B-Instruct      Qwen2.5-3B-Instruct"
    "l3b  mmlu_pro Llama-3.2-3B-Instruct    Llama-3.2-3B-Instruct"
    "l3b  medmcqa  Llama-3.2-3B-Instruct    Llama-3.2-3B-Instruct"
    "l8b  mmlu_pro Llama-3.1-8B-Instruct    Llama-3.1-8B-Instruct"
    "l8b  medmcqa  Llama-3.1-8B-Instruct    Llama-3.1-8B-Instruct"
    "nemo gsm8k    Mistral-Nemo-Instruct-2407 Mistral-Nemo-Instruct-2407"
    "nemo math     Mistral-Nemo-Instruct-2407 Mistral-Nemo-Instruct-2407"
    "nemo mmlu_pro Mistral-Nemo-Instruct-2407 Mistral-Nemo-Instruct-2407"
    "nemo medmcqa  Mistral-Nemo-Instruct-2407 Mistral-Nemo-Instruct-2407"
)

GRAFTS=(zero random random_unit shuffled_act bos_token average_act prev_layer)

# First 70 tasks: 10 cells x 7 grafts.
# Tasks 70..75: missing greedy baselines (gB) for 6 cells without existing baseline_*.
GB_CELLS=(
    "l3b  medmcqa  Llama-3.2-3B-Instruct      Llama-3.2-3B-Instruct"
    "l8b  medmcqa  Llama-3.1-8B-Instruct      Llama-3.1-8B-Instruct"
    "nemo gsm8k    Mistral-Nemo-Instruct-2407 Mistral-Nemo-Instruct-2407"
    "nemo math     Mistral-Nemo-Instruct-2407 Mistral-Nemo-Instruct-2407"
    "nemo mmlu_pro Mistral-Nemo-Instruct-2407 Mistral-Nemo-Instruct-2407"
    "nemo medmcqa  Mistral-Nemo-Instruct-2407 Mistral-Nemo-Instruct-2407"
)

if [ "$SLURM_ARRAY_TASK_ID" -lt 70 ]; then
    CELL_IDX=$(( SLURM_ARRAY_TASK_ID / 7 ))
    GRAFT_IDX=$(( SLURM_ARRAY_TASK_ID % 7 ))
    read MODEL BENCH MODEL_SUFFIX MODEL_NAME <<< "${CELLS[$CELL_IDX]}"
    SRC="${GRAFTS[$GRAFT_IDX]}"

    ANCHOR_VAR="ANCHOR_${BENCH}"
    ANCHOR="${!ANCHOR_VAR}"
    MODEL_PATH="${MODELS_DIR}/$MODEL_SUFFIX"
    OUT_DIR="results_conf/${BENCH}_ac_${SRC}_last_L26_${MODEL_NAME}_greedy_1000"

    echo "=== Task $SLURM_ARRAY_TASK_ID: $MODEL/$BENCH/graft=$SRC ==="
    echo "ANCHOR=$ANCHOR"; echo "OUT_DIR=$OUT_DIR"; date

    python experiments/run_paper_replication.py \
        --experiments ac \
        --model-a "$SRC" --model-b "$MODEL_PATH" \
        --layer-a 26 --layer-b 26 --graft-position last \
        --benchmark "$BENCH" --n-samples 1000 --seed 42 \
        --match-samples-from "$ANCHOR" \
        --output-dir "$OUT_DIR"
else
    GB_IDX=$(( SLURM_ARRAY_TASK_ID - 70 ))
    read MODEL BENCH MODEL_SUFFIX MODEL_NAME <<< "${GB_CELLS[$GB_IDX]}"

    ANCHOR_VAR="ANCHOR_${BENCH}"
    ANCHOR="${!ANCHOR_VAR}"
    MODEL_PATH="${MODELS_DIR}/$MODEL_SUFFIX"
    OUT_DIR="results_conf/${BENCH}_baseline_${MODEL_NAME}_greedy_1000"

    echo "=== Task $SLURM_ARRAY_TASK_ID: $MODEL/$BENCH greedy baseline (gB) ==="
    echo "ANCHOR=$ANCHOR"; echo "OUT_DIR=$OUT_DIR"; date

    python experiments/run_paper_replication.py \
        --experiments baselines \
        --model-a "$MODEL_PATH" --model-b "$MODEL_PATH" \
        --baseline-models a \
        --benchmark "$BENCH" --n-samples 1000 --seed 42 \
        --match-samples-from "$ANCHOR" \
        --output-dir "$OUT_DIR"
fi

date
