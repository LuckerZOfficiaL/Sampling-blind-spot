#!/bin/bash
#SBATCH --job-name=scaleup_p2
#SBATCH --time=12:00:00
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=logs/scaleup_p2_%A_%a.out
#SBATCH --error=logs/scaleup_p2_%A_%a.err
#SBATCH --array=0-59

# Template SLURM submission script. Edit:
#   * --account / --partition / --qos for your cluster
#   * env_setup.sh path (module load python, activate venv, etc.)
#   * MODELS_DIR env var (location of HuggingFace model checkpoints)
#   * ANCHOR_* paths below if you have precomputed greedy-baseline runs
#     for matched sample-ID enforcement; otherwise drop --match-samples-from

# Phase 2: 6 sampling seeds (42..47) per new cell. 10 cells = 60 tasks.
# Temperature 0.7, top-p 0.9. IDs matched to the same benchmark-level anchor as phase 1.

# Source your environment-setup script (module load, venv activate, etc.):
# source /path/to/your/env_setup.sh
cd "$(dirname "$0")/.."  # repo root
ANCHOR_gsm8k="results_conf/gsm8k_baseline_Qwen2.5-3B-Instruct_greedy_1000/baseline_Qwen2_5_3B_Instruct/predictions.json"
ANCHOR_math="results_conf/math_baseline_Qwen2.5-3B-Instruct_greedy_1000/baseline_Qwen2_5_3B_Instruct/predictions.json"
ANCHOR_mmlu_pro="results_greedy_decoding/mmlu_pro_ac_zero_llama8B_1000/ac_trained_projection/predictions.json"
ANCHOR_medmcqa="results_no_truncation/medmcqa_ac_random_qwen3b_1000/ac_trained_projection/predictions.json"

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

SEEDS=(42 43 44 45 46 47)

CELL_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 6 ))
read MODEL BENCH MODEL_SUFFIX MODEL_NAME <<< "${CELLS[$CELL_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

ANCHOR_VAR="ANCHOR_${BENCH}"
ANCHOR="${!ANCHOR_VAR}"

MODEL_PATH="${MODELS_DIR}/$MODEL_SUFFIX"
OUT_DIR="results_sampling/${BENCH}_baseline_${MODEL_NAME}_sampling_seed${SEED}_1000"

echo "=== Task $SLURM_ARRAY_TASK_ID: $MODEL/$BENCH/seed=$SEED ==="
echo "ANCHOR=$ANCHOR"
echo "OUT_DIR=$OUT_DIR"
date

python experiments/run_paper_replication.py \
    --experiments baselines \
    --model-a "$MODEL_PATH" --model-b "$MODEL_PATH" \
    --baseline-models a \
    --benchmark "$BENCH" --n-samples 1000 --seed "$SEED" \
    --do-sample --temperature 0.7 --top-p 0.9 \
    --match-samples-from "$ANCHOR" \
    --output-dir "$OUT_DIR"

date
