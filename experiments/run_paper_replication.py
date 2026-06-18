#!/usr/bin/env python3
# pylint: disable=redefined-outer-name,too-many-lines,wrong-import-position
"""
Paper Replication: "Communicating Activations Between Language Model Agents"
(arXiv:2501.14082)

This script replicates the GSM8K experiments from Table 1:
- Baselines: 3.2-3B and 3.1-8B alone
- AC (Activation Communication): sum, replace, mean
- AC (W): Learned projection matrix variants

Paper methodology:
- Uses instruct-tuned models (Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct)
- Nucleus sampling with p=0.9
- 100-sample subset of GSM8K
- Bootstrap CIs with 1000 iterations

Usage:
    python experiments/run_paper_replication.py --experiments baselines
    python experiments/run_paper_replication.py --experiments ac
    python experiments/run_paper_replication.py --experiments all
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.benchmarks.reasoning_tasks import (
    load_gsm8k_subset,
    MMLUExample, load_mmlu_subset, load_mmlu_pro_subset,
    MATHExample, load_math_subset,
    load_svamp_subset,
    GSMPlusExample, load_gsm_plus_subset,
    load_gsm_symbolic_subset,
    load_codemmlu_subset,
    load_rbi_subset,
    load_medmcqa_subset,
    load_wikimia_subset,
    load_numinamath_tir_subset,
)
from src.communication.activation_graft import ActivationGraftingEngine, GraftingMode
from src.communication.combination_functions import get_combination_function
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.metrics import MetricTracker, numerical_match
from src.models.activation_extractor import ExtractionPoint
from src.models.layer_utils import get_transformer_layers, unpack_hook_output
from src.models.model_loader import load_model_and_tokenizer, get_model_info
from src.models.model_registry import check_dimension_compatibility
from src.utils.config import ModelConfig, set_seed
from src.utils.wandb_utils import WandbRun, init_wandb, timer

# Modal is only needed for cloud execution, not local runs
# import modal
#
# app = modal.App("paper-replication")
#
# modal_image = (
#     modal.Image.debian_slim(python_version="3.11")
#     .pip_install(
#         "torch",
#         "transformers",
#         "accelerate",
#         "huggingface_hub",
#         "datasets",
#         "numpy",
#         "wandb",
#         "pyyaml",
#         "tqdm",
#     )
#     .add_local_dir("src", remote_path="/root/src")
#     .add_local_dir("projections", remote_path="/root/projections")
# )

# =============================================================================
# PAPER REPLICATION CONFIGURATION - Single source of truth
# =============================================================================
# Based on arXiv:2501.14082 "Communicating Activations Between Language Model Agents"

# Layer configuration (middle layers for semantic information transfer)
PAPER_CONFIG = {
    "layer_a": 26,           # Layer to extract from source model
    "layer_b": 26,           # Layer to inject into target model
    "projection_path": "projections_26-26/w_3b_to_8b.pt",
    #"projection_path": "/projections/w_Deepsync3b_to_8b.pt",

    # Single combination function:
    "combination_fn": "trained_projection",

    # To compare multiple, change to a list (will run all and save separate results):
    # "combination_fn": ["replace", "sum", "mean", "weighted_sum" "trained_projection", "learned_linear", "concat_project", "gated"] where all but "replace" need training some module
}

# Model paths - automatically detect local vs HuggingFace
if os.getenv("TRANSFORMERS_OFFLINE") == "1":
    # Check for local model directories (Instruct versions)
    _ckpts_dir = os.getenv("TRANSFORMERS_CACHE") or os.environ.get("MODELS_DIR", "")
    _model_a_local = os.path.join(_ckpts_dir, "Llama-3.2-3B-Instruct")
    _model_b_local = os.path.join(_ckpts_dir, "Llama-3.1-8B-Instruct")

    if os.path.exists(_model_a_local) and os.path.exists(_model_b_local):
        MODEL_A_NAME = _model_a_local
        MODEL_B_NAME = _model_b_local
        print(f"[CONFIG] Using local Instruct models: {MODEL_A_NAME}, {MODEL_B_NAME}")
    else:
        # Fallback: try base models
        _model_a_base = os.path.join(_ckpts_dir, "Llama-3.2-3B")
        _model_b_base = os.path.join(_ckpts_dir, "Llama-3.1-8B")
        if os.path.exists(_model_a_base) and os.path.exists(_model_b_base):
            MODEL_A_NAME = _model_a_base
            MODEL_B_NAME = _model_b_base
            print(f"[CONFIG] WARNING: Using base models (not Instruct). Results will differ from paper.")
        else:
            MODEL_A_NAME = "meta-llama/Llama-3.2-3B-Instruct"
            MODEL_B_NAME = "meta-llama/Llama-3.1-8B-Instruct"
            print(f"[CONFIG] WARNING: Local models not found, will try HuggingFace names (may fail offline)")
else:
    MODEL_A_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    MODEL_B_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"[CONFIG] Using HuggingFace model names (online mode)")

print(f"[CONFIG] Paper configuration: layer_a={PAPER_CONFIG['layer_a']}, layer_b={PAPER_CONFIG['layer_b']}, combination={PAPER_CONFIG['combination_fn']}")


def load_sample_ids_from_predictions(predictions_path: str) -> List[str]:
    """
    Load sample IDs from a predictions.json file.

    This allows running experiments with the exact same samples as a previous run.

    Args:
        predictions_path: Path to predictions.json from a previous experiment

    Returns:
        List of sample IDs (e.g., ['gsm8k_1309', 'gsm8k_0228', ...])
    """
    with open(predictions_path, 'r', encoding="utf-8") as f:
        predictions = json.load(f)

    sample_ids = [p['id'] for p in predictions]
    return sample_ids


def filter_examples_by_ids(examples: List, sample_ids: List[str]) -> List:
    """
    Filter examples to only include those with IDs in sample_ids.

    Maintains the order of sample_ids.

    Args:
        examples: Full list of examples
        sample_ids: List of IDs to keep

    Returns:
        Filtered list of examples in the order of sample_ids
    """
    # Create lookup by ID
    id_to_example = {ex.id: ex for ex in examples}

    # Filter and maintain order
    filtered = []
    missing = []
    for sid in sample_ids:
        if sid in id_to_example:
            filtered.append(id_to_example[sid])
        else:
            missing.append(sid)

    if missing:
        print(f"Warning: {len(missing)} sample IDs not found in dataset: {missing[:5]}...")

    return filtered


def simple_bootstrap_ci(
    correct_list: List[int],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for accuracy from a list of binary correctness values.

    Args:
        correct_list: List of 1s (correct) and 0s (incorrect)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed

    Returns:
        Tuple of (lower_bound, upper_bound) as percentages
    """
    np.random.seed(seed)
    n = len(correct_list)
    correct_array = np.array(correct_list)

    bootstrap_accuracies = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        sample = correct_array[indices]
        accuracy = sample.mean() * 100
        bootstrap_accuracies.append(accuracy)

    bootstrap_accuracies = np.array(bootstrap_accuracies)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_accuracies, 100 * alpha / 2)
    upper = np.percentile(bootstrap_accuracies, 100 * (1 - alpha / 2))

    return lower, upper


# Paper's generation config: nucleus sampling with p=0.9
GENERATION_CONFIG = {
    "do_sample": False,
    "top_p": 0.9, # ignored when do_sample = False
    "temperature": 0.7, # ignored when do_sample = False
    "max_new_tokens": 2048,
}


def format_chat_prompt(question: str, tokenizer) -> str:
    """Format question using the model's chat template."""
    messages = [
        {
            "role": "user",
            "content": (
                "Solve this math problem step by step. "
                "At the end, provide your final answer "
                "as a number after '#### '.\n\n"
                f"Problem: {question}"
            ),
        }
    ]

    # Use the tokenizer's chat template if available
    if getattr(tokenizer, 'chat_template', None) is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    # Fallback for models without chat template
    return (
        f"Problem: {question}\n\n"
        "Solve step by step and end with #### "
        "followed by the numerical answer:\n"
    )


def extract_answer_from_response(response: str) -> Optional[str]:
    """
    Extract numerical answer from model response.
    Looks for <answer> tag first, then #### pattern, then falls back to last number.
    """
    # Look for <answer>...</answer> tag (reasoning model format, e.g. yakuraku)
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.IGNORECASE | re.DOTALL)
    if match:
        inner = match.group(1)
        # Extract number from inside the tag (may contain #### prefix)
        num = re.search(r'-?[\d,]+\.?\d*', inner)
        if num:
            return num.group(0).replace(',', '')

    # Look for #### pattern (GSM8K standard format)
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', response)
    if match:
        return match.group(1).replace(',', '')

    # Look for "answer is" pattern
    match = re.search(
        r'(?:answer|result|total)\s*(?:is|=|:)\s*\$?(-?[\d,]+\.?\d*)',
        response,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).replace(',', '')

    # Fallback: find the last number in the response
    numbers = re.findall(r'-?[\d,]+\.?\d*', response)
    if numbers:
        # Filter out very small numbers that are likely not answers
        valid_numbers = [
            n.replace(',', '') for n in numbers
            if len(n.replace(',', '').replace('.', '')
                    .replace('-', '')) > 0
        ]
        if valid_numbers:
            return valid_numbers[-1]

    return None


_MMLU_CHOICE_LABELS = ["A", "B", "C", "D"]


def format_mmlu_prompt(example: MMLUExample, tokenizer) -> str:
    """Format an MMLU question using the model's chat template."""
    choices_text = "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(_MMLU_CHOICE_LABELS, example.choices)
    )
    content = (
        "The following is a multiple choice question. "
        "Think step by step, then wrap your final answer as <answer>X</answer> "
        "where X is A, B, C, or D.\n\n"
        f"Question: {example.question}\n"
        f"{choices_text}\n\n"
        "Answer:"
    )
    messages = [{"role": "user", "content": content}]
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return content


def extract_mmlu_answer(response: str) -> Optional[str]:
    """Extract choice letter (A/B/C/D) from model response."""
    # Prefer explicit tag: <answer>X</answer>
    match = re.search(r"<answer>\s*([ABCD])\s*</answer>", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: bare letter at the start
    match = re.match(r"^\s*([ABCD])\b", response.strip())
    if match:
        return match.group(1)
    match = re.search(
        r"(?:answer|option)\s*(?:is|:)\s*([ABCD])\b", response, re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    # Use the LAST standalone letter to avoid matching letters in explanations
    # like "Comparing A with B, option D is correct" → return "D", not "A".
    all_matches = re.findall(r"\b([ABCD])\b", response)
    if all_matches:
        return all_matches[-1].upper()
    return None


def check_mmlu_correct(predicted: Optional[str], expected: str) -> bool:
    """Check whether predicted letter matches expected letter."""
    if predicted is None:
        return False
    return predicted.upper() == expected.upper()


# Threshold for token-F1 "correct" on RBI open-ended QA
_RBI_F1_THRESHOLD = 0.3

from src.evaluation.metrics import f1_score as _token_f1  # pylint: disable=wrong-import-position


def compute_rbi_f1(generated: str, reference: str) -> float:
    """Return token-level F1 between generated text and the reference answer."""
    return _token_f1(generated, reference)


def check_rbi_correct(generated: str, reference: str) -> bool:
    """Return True when token-F1 >= _RBI_F1_THRESHOLD."""
    return compute_rbi_f1(generated, reference) >= _RBI_F1_THRESHOLD


_MMLU_PRO_CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def format_mmlu_pro_prompt(example: MMLUExample, tokenizer) -> str:
    """Format an MMLU-Pro question (10 choices, A-J) using the model's chat template."""
    labels = _MMLU_PRO_CHOICE_LABELS[:len(example.choices)]
    choices_text = "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(labels, example.choices)
    )
    content = (
        "The following is a multiple choice question. "
        "Think step by step, then wrap your final answer as <answer>X</answer> "
        "where X is one of A, B, C, D, E, F, G, H, I, or J.\n\n"
        f"Question: {example.question}\n"
        f"{choices_text}\n\n"
        "Answer:"
    )
    messages = [{"role": "user", "content": content}]
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return content


def extract_mmlu_pro_answer(response: str) -> Optional[str]:
    """Extract choice letter (A-J) from model response."""
    match = re.search(r"<answer>\s*([A-J])\s*</answer>", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.match(r"^\s*([A-J])\b", response.strip())
    if match:
        return match.group(1).upper()
    match = re.search(
        r"(?:answer|option)\s*(?:is|:)\s*([A-J])\b", response, re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    all_matches = re.findall(r"\b([A-J])\b", response)
    if all_matches:
        return all_matches[-1].upper()
    return None


def format_math_prompt(example: MATHExample, tokenizer) -> str:
    """Format a Hendrycks MATH question using the model's chat template."""
    content = (
        "Solve the following competition math problem step by step. "
        "Put your final answer inside \\boxed{}.\n\n"
        f"Problem: {example.question}"
    )
    messages = [{"role": "user", "content": content}]
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return content


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content of the last \\boxed{...} in text, handling nested braces."""
    tag = r"\boxed{"
    idx = text.rfind(tag)
    if idx == -1:
        return None
    start = idx + len(tag)
    depth = 1
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return text[start:].strip()


def _normalize_math_answer(s: str) -> str:
    """Normalize a MATH answer string for comparison."""
    s = s.strip()
    # Remove surrounding dollar signs
    s = s.strip("$").strip()
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def extract_math_answer(response: str) -> Optional[str]:
    """Extract the final answer from a MATH model response."""
    boxed = _extract_boxed(response)
    if boxed is not None:
        return boxed
    # Fallback: look for "answer is X" pattern
    match = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*(.+?)(?:\.|$)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def check_math_correct(predicted: Optional[str], expected: str) -> bool:
    """Check whether predicted MATH answer matches expected (normalized string comparison)."""
    if predicted is None:
        return False
    return _normalize_math_answer(predicted) == _normalize_math_answer(expected)


def setup_experiment_logging(output_dir: Path, experiment_name: str) -> logging.Logger:
    """Setup logging for an experiment to both file and console."""
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # File handler
    log_file = output_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def get_device_info() -> Tuple[str, str]:
    """Get device and detailed device info."""
    if torch.backends.mps.is_available():
        device = "mps"
        device_info = "Apple Silicon GPU (MPS)"
    elif torch.cuda.is_available():
        device = "cuda"
        device_info = f"CUDA: {torch.cuda.get_device_name(0)}"
    else:
        device = "cpu"
        device_info = "CPU"
    return device, device_info


def load_model_with_info(
    config: ModelConfig,
    device: str,
    logger: logging.Logger,
) -> Tuple[Any, Any]:
    """Load model and print device info."""
    logger.info("  Loading %s...", config.name)
    model, tokenizer = load_model_and_tokenizer(config, device)

    info = get_model_info(model)
    logger.info("    Device: %s", info['device'])
    logger.info("    Dtype: %s", info['dtype'])
    logger.info("    Parameters: %s", f"{info['num_parameters']:,}")
    logger.info("    Layers: %s, d_model: %s", info['n_layers'], info['d_model'])

    return model, tokenizer


def _compute_output_confidence(scores_tuple, generated_ids) -> Dict[str, float]:
    """
    Compute per-output-token confidence metrics from generation scores.

    Args:
        scores_tuple: Tuple of (vocab_size,) logit tensors, one per generated token.
                      From model.generate(output_scores=True, return_dict_in_generate=True).
        generated_ids: 1-D tensor of generated token IDs (prompt stripped).

    Returns:
        Dict with keys:
            mean_token_confidence — mean softmax probability of the chosen token
            mean_token_log_prob   — mean log probability of the chosen token
            seq_log_prob          — sum of log probabilities of chosen tokens
    """
    if scores_tuple is None or len(scores_tuple) == 0:
        return {"mean_token_confidence": None, "mean_token_log_prob": None, "seq_log_prob": None}

    token_probs = []
    for t, logits in enumerate(scores_tuple):
        if t >= len(generated_ids):
            break
        probs = torch.softmax(logits[0].float(), dim=-1)
        chosen_id = generated_ids[t].item()
        token_probs.append(probs[chosen_id].item())

    if not token_probs:
        return {"mean_token_confidence": None, "mean_token_log_prob": None, "seq_log_prob": None}

    import math
    log_probs = [math.log(max(p, 1e-45)) for p in token_probs]
    return {
        "mean_token_confidence": sum(token_probs) / len(token_probs),
        "mean_token_log_prob": sum(log_probs) / len(log_probs),
        "seq_log_prob": sum(log_probs),
    }


def _compute_input_confidence(model, tokenizer, prompt: str) -> Dict[str, float]:
    """
    Compute per-input-token confidence metrics via a forward pass on the prompt.

    For each token position i (1..n), computes P(token_i | token_0..token_{i-1})
    from the model logits — i.e. prompt perplexity in probability space.

    Returns:
        Dict with keys:
            input_mean_token_confidence — mean softmax probability of each prompt token
            input_mean_token_log_prob   — mean log probability of each prompt token
            input_seq_log_prob          — sum of log probabilities of prompt tokens
    """
    import math
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"][0]  # (seq_len,)
    if len(input_ids) < 2:
        return {"input_mean_token_confidence": None, "input_mean_token_log_prob": None, "input_seq_log_prob": None}
    with torch.no_grad():
        logits = model(**inputs).logits[0]  # (seq_len, vocab)
    token_probs = []
    for i in range(1, len(input_ids)):
        probs = torch.softmax(logits[i - 1].float(), dim=-1)
        token_probs.append(probs[input_ids[i].item()].item())
    log_probs = [math.log(max(p, 1e-45)) for p in token_probs]
    return {
        "input_mean_token_confidence": sum(token_probs) / len(token_probs),
        "input_mean_token_log_prob": sum(log_probs) / len(log_probs),
        "input_seq_log_prob": sum(log_probs),
    }


def run_baseline_evaluation(
    model,
    tokenizer,
    examples: List,
    benchmark: str = "gsm8k",
    model_name: str = "model",
    _logger: logging.Logger = None,
    wb_run: WandbRun = None,
    log_first_token_entropy: bool = False,
    save_b_activations: bool = False,
    activations_dir: Path = None,
    activation_layer: int = 26,
    save_output_confidence: bool = False,
    save_input_confidence: bool = False,
    input_only: bool = False,
    save_token_scores: bool = False,
) -> Tuple[MetricTracker, ErrorAnalyzer, List[Dict]]:
    """
    Run baseline evaluation (single model, no grafting).
    Uses paper's methodology: nucleus sampling p=0.9, chat format.

    If input_only=True, skips generation and only collects input confidence metrics
    via a single forward pass on the prompt. generated='', correct=None.

    Returns:
        Tuple of (tracker, analyzer, predictions)
    """
    if input_only:
        save_input_confidence = True
    if wb_run is None:
        wb_run = WandbRun()
    tracker = MetricTracker()
    analyzer = ErrorAnalyzer()
    predictions = []

    for idx, ex in enumerate(tqdm(examples, desc=f"Evaluating {model_name}")):
        # Format prompt based on benchmark
        if benchmark in ("mmlu", "codemmlu", "medmcqa"):
            prompt = format_mmlu_prompt(ex, tokenizer)
        elif benchmark == "mmlu_pro":
            prompt = format_mmlu_pro_prompt(ex, tokenizer)
        elif benchmark == "math":
            prompt = format_math_prompt(ex, tokenizer)
        elif benchmark == "wikimia":
            prompt = ""  # text is entirely in ex.answer; scored raw with no chat template
        else:
            prompt = format_chat_prompt(ex.question, tokenizer)

        # In input_only mode, append ground truth answer to measure P(question + answer)
        if input_only:
            gt = ex.expected if hasattr(ex, 'expected') else ex.answer
            separator = "" if not prompt else "\n"
            prompt = prompt + separator + str(gt)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Capture layer activation for probe training if requested
        if save_b_activations and activations_dir is not None:
            _captured = {}
            _layers = get_transformer_layers(model)

            def _capture_hook(module, inp, output):  # pylint: disable=unused-argument
                hidden, _ = unpack_hook_output(output)
                _captured['act'] = hidden[0, -1, :].detach().cpu()

            _hook = _layers[activation_layer].register_forward_hook(_capture_hook)
            with torch.no_grad():
                model(**inputs)
            _hook.remove()
            safe_id = str(ex.id).replace('/', '_').replace(' ', '_')
            torch.save(_captured['act'], activations_dir / f"{safe_id}.pt")

        sample_start = time.time()
        output_confidence = None
        input_confidence = None
        first_token_entropy = None
        token_scores = None
        generated = ""
        is_correct = None
        extracted_answer = None

        if input_only:
            # Fast path: single forward pass on prompt, no generation
            with torch.no_grad():
                logits = model(**inputs).logits[0]  # (prompt_len, vocab)
            sample_elapsed = time.time() - sample_start
            import math as _math
            _input_ids = inputs['input_ids'][0]
            _token_probs = []
            for _i in range(1, len(_input_ids)):
                _probs = torch.softmax(logits[_i - 1].float(), dim=-1)
                _token_probs.append(_probs[_input_ids[_i].item()].item())
            _log_probs = [_math.log(max(p, 1e-45)) for p in _token_probs]
            input_confidence = {
                "input_mean_token_confidence": sum(_token_probs) / len(_token_probs),
                "input_mean_token_log_prob": sum(_log_probs) / len(_log_probs),
                "input_seq_log_prob": sum(_log_probs),
            }
            if save_token_scores:
                import torch.nn.functional as _F
                # Vectorised: compute per-token log-probs and Min-K++ z-scores in one pass
                _logits_slice = logits[:-1].float()          # (T-1, V)
                _probs_full   = _F.softmax(_logits_slice, dim=-1)
                _logp_full    = _F.log_softmax(_logits_slice, dim=-1)
                _tok_ids      = _input_ids[1:].unsqueeze(1)  # (T-1, 1)
                _tok_logp     = _logp_full.gather(1, _tok_ids).squeeze(1)  # (T-1,)
                _mu           = (_probs_full * _logp_full).sum(-1)          # (T-1,)
                _sigma        = (
                    (_probs_full * _logp_full.pow(2)).sum(-1) - _mu.pow(2)
                ).clamp(min=1e-8).sqrt()                                    # (T-1,)
                token_scores = {
                    "token_log_probs":     _tok_logp.tolist(),
                    "token_minkpp_scores": ((_tok_logp - _mu) / _sigma).tolist(),
                }
                del _logits_slice, _probs_full, _logp_full
            del logits  # free the full logit matrix from GPU memory
        else:
            need_scores = log_first_token_entropy or save_output_confidence

            # Capture prefill logits from inside generate() to avoid a separate forward pass
            _prefill_logits = [None]
            _lm_head_hook = None
            if save_input_confidence:
                def _capture_prefill(module, inp, out):  # pylint: disable=unused-argument
                    if _prefill_logits[0] is None:  # only capture the first (prefill) call
                        _prefill_logits[0] = out.detach()
                _lm_head_hook = model.lm_head.register_forward_hook(_capture_prefill)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                    do_sample=GENERATION_CONFIG["do_sample"],
                    top_p=GENERATION_CONFIG["top_p"],
                    temperature=GENERATION_CONFIG["temperature"],
                    pad_token_id=tokenizer.pad_token_id,
                    output_scores=need_scores,
                    return_dict_in_generate=need_scores,
                )

            if _lm_head_hook is not None:
                _lm_head_hook.remove()
            sample_elapsed = time.time() - sample_start

            # Extract scores and sequences when return_dict_in_generate=True
            output_scores_tuple = None
            if need_scores:
                output_scores_tuple = outputs.scores
                if log_first_token_entropy:
                    first_logits = outputs.scores[0][0].float()  # (vocab_size,)
                    first_probs = torch.softmax(first_logits, dim=-1)
                    first_token_entropy = float(-torch.sum(first_probs * torch.log(first_probs + 1e-10)))
                outputs = outputs.sequences

            input_len = inputs['input_ids'].shape[1]

            # Decode only the new tokens
            generated = tokenizer.decode(
                outputs[0][input_len:],
                skip_special_tokens=True
            ).strip()

            # Compute output confidence metrics if requested
            if save_output_confidence:
                output_confidence = _compute_output_confidence(output_scores_tuple, outputs[0][input_len:])

            # Compute input confidence metrics from captured prefill logits (no extra forward pass)
            if save_input_confidence and _prefill_logits[0] is not None:
                import math as _math
                prefill_logits = _prefill_logits[0][0]  # (prompt_len, vocab)
                _input_ids = inputs['input_ids'][0]
                _token_probs = []
                for _i in range(1, len(_input_ids)):
                    _probs = torch.softmax(prefill_logits[_i - 1].float(), dim=-1)
                    _token_probs.append(_probs[_input_ids[_i].item()].item())
                _log_probs = [_math.log(max(p, 1e-45)) for p in _token_probs]
                input_confidence = {
                    "input_mean_token_confidence": sum(_token_probs) / len(_token_probs),
                    "input_mean_token_log_prob": sum(_log_probs) / len(_log_probs),
                    "input_seq_log_prob": sum(_log_probs),
                }
                del _prefill_logits[0]  # free memory

        # Extract answer and check correctness based on benchmark
        if input_only:
            expected = ex.expected if hasattr(ex, 'expected') else ex.answer
            rbi_f1 = None
        elif benchmark in ("mmlu", "codemmlu", "medmcqa"):
            extracted_answer = extract_mmlu_answer(generated)
            is_correct = check_mmlu_correct(extracted_answer, ex.expected)
            expected = ex.expected
        elif benchmark == "mmlu_pro":
            extracted_answer = extract_mmlu_pro_answer(generated)
            is_correct = check_mmlu_correct(extracted_answer, ex.expected)
            expected = ex.expected
        elif benchmark == "math":
            extracted_answer = extract_math_answer(generated)
            is_correct = check_math_correct(extracted_answer, ex.answer)
            expected = ex.answer
        elif benchmark == "rbi":
            extracted_answer = generated
            rbi_f1 = compute_rbi_f1(generated, ex.answer)
            is_correct = rbi_f1 >= _RBI_F1_THRESHOLD
            expected = ex.answer
        else:
            extracted_answer = extract_answer_from_response(generated)
            is_correct = (
                numerical_match(extracted_answer or "", ex.answer)
                if extracted_answer else False
            )
            expected = ex.answer

        # Track metrics (skip in input_only mode — no predictions to evaluate)
        if not input_only:
            tracker.add(extracted_answer or "", expected, ex.id)

        # Log per-sample metrics to W&B
        n_correct_so_far = sum(1 for p in predictions if p.get("correct")) + int(is_correct or 0)
        wb_run.log({
            "sample/correct": int(is_correct or 0),
            "sample/running_accuracy": n_correct_so_far / (idx + 1),
            "sample/generation_time_s": sample_elapsed,
        }, step=idx)

        # Log for error analysis
        analyzer.add(
            task_id=ex.id,
            benchmark=benchmark,
            input_a=prompt,
            input_b="",
            expected=expected,
            predicted=extracted_answer or generated[:100],
            correct=is_correct,
            layer_a=-1,
            layer_b=-1,
            combination_fn="baseline",
            grafting_mode="none",
        )

        pred = {
            "id": ex.id,
            "question": ex.question,
            "expected": expected,
            "generated": generated,
            "extracted_answer": extracted_answer,
            "correct": is_correct,
        }
        if log_first_token_entropy:
            pred["first_token_entropy"] = first_token_entropy
        if output_confidence is not None:
            pred.update(output_confidence)
        if input_confidence is not None:
            pred.update(input_confidence)
        if token_scores is not None:
            pred.update(token_scores)
        if benchmark in ("mmlu", "mmlu_pro", "codemmlu", "medmcqa"):
            pred["subject"] = ex.subject
            pred["choices"] = ex.choices
        elif benchmark == "math":
            pred["subject"] = ex.subject
            pred["level"] = ex.level
        elif benchmark == "rbi":
            pred["f1"] = rbi_f1
        elif benchmark == "wikimia":
            pred["wikimia_label"] = ex.label
        predictions.append(pred)

        if len(predictions) % 10 == 0:
            n_c = sum(1 for p in predictions if p["correct"])
            print(f"  [{len(predictions)}] acc={n_c}/{len(predictions)}"
                  f" ({100*n_c/len(predictions):.1f}%)")

    return tracker, analyzer, predictions


def run_grafting_evaluation(
    engine: ActivationGraftingEngine,
    examples: List,
    tokenizer_a,
    tokenizer_b,
    combination_fn_name: str,
    benchmark: str = "gsm8k",
    experiment_name: str = "grafting",
    _logger: logging.Logger = None,
    wb_run: WandbRun = None,
    log_first_token_entropy: bool = False,
    save_output_confidence: bool = False,
    save_input_confidence: bool = False,
    input_only: bool = False,
) -> Tuple[MetricTracker, ErrorAnalyzer, List[Dict]]:
    """
    Run grafting evaluation with paper's methodology.

    If input_only=True, skips generation and runs only a single grafted forward pass
    to collect input confidence metrics. prompt_b is extended with the ground truth answer.

    Returns:
        Tuple of (tracker, analyzer, predictions)
    """
    if input_only:
        save_input_confidence = True
    if wb_run is None:
        wb_run = WandbRun()
    tracker = MetricTracker()
    analyzer = ErrorAnalyzer()
    predictions = []

    for idx, ex in enumerate(tqdm(examples, desc=f"Evaluating {experiment_name}")):
        # Format prompts based on benchmark
        if benchmark in ("mmlu", "codemmlu", "medmcqa"):
            prompt_a = "" if tokenizer_a is None else format_mmlu_prompt(ex, tokenizer_a)
            prompt_b = format_mmlu_prompt(ex, tokenizer_b)
        elif benchmark == "mmlu_pro":
            prompt_a = "" if tokenizer_a is None else format_mmlu_pro_prompt(ex, tokenizer_a)
            prompt_b = format_mmlu_pro_prompt(ex, tokenizer_b)
        elif benchmark == "math":
            prompt_a = "" if tokenizer_a is None else format_math_prompt(ex, tokenizer_a)
            prompt_b = format_math_prompt(ex, tokenizer_b)
        elif benchmark == "wikimia":
            prompt_a = ""  # text is the full input; no chat template
            prompt_b = ""
        else:
            prompt_a = "" if tokenizer_a is None else format_chat_prompt(ex.question, tokenizer_a)
            prompt_b = format_chat_prompt(ex.question, tokenizer_b)

        # In input_only mode, append ground truth to prompt_b before grafted forward pass
        if input_only:
            gt = ex.expected if hasattr(ex, 'expected') else ex.answer
            separator = "" if not prompt_b else "\n"
            prompt_b_for_fwd = prompt_b + separator + str(gt)
        else:
            prompt_b_for_fwd = prompt_b

        sample_start = time.time()
        output_confidence = None
        input_confidence = None
        output = ""
        is_correct = None
        extracted_answer = None
        diagnostics = None

        if input_only:
            # Fast path: single grafted forward pass, no generation.
            # Replicate engine.forward_with_graft but keep full logit matrix.
            activation_a, _ = engine._extract_from_a(prompt_a)
            inputs_b_full = tokenizer_b(prompt_b_for_fwd, return_tensors="pt").to(engine.model_b.device)
            hook = engine._create_grafting_hook(activation_a, engine.graft_position)
            target_layer = engine._get_target_layer()
            hook_handle = target_layer.register_forward_hook(hook)
            if hasattr(engine.combination_fn, 'make_capture_hook'):
                _layers = get_transformer_layers(engine.model_b)
                _prev = _layers[max(0, engine.layer_b - 1)]
                _cap_handle = _prev.register_forward_hook(
                    engine.combination_fn.make_capture_hook(engine.graft_position)
                )
            else:
                _cap_handle = None
            try:
                with torch.no_grad():
                    _outputs = engine.model_b(**inputs_b_full)
            finally:
                hook_handle.remove()
                if _cap_handle is not None:
                    _cap_handle.remove()
            logits_full = _outputs.logits[0]  # (seq_len, vocab)
            diagnostics = None
            sample_elapsed = time.time() - sample_start
            import math as _math
            inputs_b_ids = inputs_b_full["input_ids"][0]
            _token_probs = []
            for _i in range(1, len(inputs_b_ids)):
                _probs = torch.softmax(logits_full[_i - 1].float(), dim=-1)
                _token_probs.append(_probs[inputs_b_ids[_i].item()].item())
            _log_probs = [_math.log(max(p, 1e-45)) for p in _token_probs]
            input_confidence = {
                "input_mean_token_confidence": sum(_token_probs) / len(_token_probs),
                "input_mean_token_log_prob": sum(_log_probs) / len(_log_probs),
                "input_seq_log_prob": sum(_log_probs),
            }
        else:
            engine_result = engine.generate(
                prompt_a=prompt_a,
                prompt_b=prompt_b_for_fwd,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                return_diagnostics=True,
                return_output_scores=save_output_confidence,
                do_sample=GENERATION_CONFIG["do_sample"],
                top_p=GENERATION_CONFIG["top_p"],
                temperature=GENERATION_CONFIG["temperature"],
            )
            sample_elapsed = time.time() - sample_start

            if save_output_confidence:
                output, diagnostics, output_scores_tuple, gen_ids = engine_result
            else:
                output, diagnostics = engine_result
                output_scores_tuple = None
                gen_ids = None
            output = output.strip()

            # Compute output confidence metrics if requested
            if save_output_confidence:
                output_confidence = _compute_output_confidence(output_scores_tuple, gen_ids)

            # Compute input confidence metrics if requested
            if save_input_confidence:
                input_confidence = _compute_input_confidence(engine.model_b, tokenizer_b, prompt_b)

        # Compute first-token entropy of model_b (pre-graft baseline uncertainty) if requested
        first_token_entropy = None
        if log_first_token_entropy:
            inputs_b = tokenizer_b(prompt_b, return_tensors="pt").to(engine.model_b.device)
            with torch.no_grad():
                logits_b = engine.model_b(**inputs_b).logits[0, -1, :].float()
            probs_b = torch.softmax(logits_b, dim=-1)
            first_token_entropy = float(-torch.sum(probs_b * torch.log(probs_b + 1e-10)))

        # Extract answer and check correctness based on benchmark
        if input_only:
            expected = ex.expected if hasattr(ex, 'expected') else ex.answer
            rbi_f1 = None
        elif benchmark in ("mmlu", "codemmlu", "medmcqa"):
            extracted_answer = extract_mmlu_answer(output)
            is_correct = check_mmlu_correct(extracted_answer, ex.expected)
            expected = ex.expected
        elif benchmark == "mmlu_pro":
            extracted_answer = extract_mmlu_pro_answer(output)
            is_correct = check_mmlu_correct(extracted_answer, ex.expected)
            expected = ex.expected
        elif benchmark == "math":
            extracted_answer = extract_math_answer(output)
            is_correct = check_math_correct(extracted_answer, ex.answer)
            expected = ex.answer
        elif benchmark == "rbi":
            extracted_answer = output
            rbi_f1 = compute_rbi_f1(output, ex.answer)
            is_correct = rbi_f1 >= _RBI_F1_THRESHOLD
            expected = ex.answer
        else:
            extracted_answer = extract_answer_from_response(output)
            is_correct = (
                numerical_match(extracted_answer or "", ex.answer)
                if extracted_answer else False
            )
            expected = ex.answer

        # Track metrics (skip in input_only mode)
        if not input_only:
            tracker.add(extracted_answer or "", expected, ex.id)

        # Log per-sample metrics and activation diagnostics to W&B
        n_correct_so_far = sum(1 for p in predictions if p.get("correct")) + int(is_correct or 0)
        wb_log = {
            "sample/correct": int(is_correct or 0),
            "sample/running_accuracy": n_correct_so_far / (idx + 1),
            "sample/generation_time_s": sample_elapsed,
        }
        if diagnostics is not None:
            wb_log.update({
                "activation/a_norm": diagnostics.activation_a_norm,
                "activation/b_norm": diagnostics.activation_b_norm,
                "activation/combined_norm": diagnostics.combined_norm,
                "activation/cosine_sim_ab": diagnostics.cosine_similarity_ab,
                "activation/output_entropy": diagnostics.output_entropy,
            })
        wb_run.log(wb_log, step=idx)

        # Log for error analysis
        analyzer.add(
            task_id=ex.id,
            benchmark=benchmark,
            input_a=prompt_a[:200],  # Truncate for storage
            input_b=prompt_b[:200],
            expected=expected,
            predicted=extracted_answer or output[:100],
            correct=is_correct,
            layer_a=engine.layer_a,
            layer_b=engine.layer_b,
            combination_fn=combination_fn_name,
            grafting_mode=engine.grafting_mode.value,
            activation_a_norm=diagnostics.activation_a_norm if diagnostics else None,
            activation_b_norm=diagnostics.activation_b_norm if diagnostics else None,
            combined_norm=diagnostics.combined_norm if diagnostics else None,
            cosine_similarity_ab=diagnostics.cosine_similarity_ab if diagnostics else None,
            output_entropy=diagnostics.output_entropy if diagnostics else None,
        )

        pred = {
            "id": ex.id,
            "question": ex.question,
            "expected": expected,
            "generated": output,
            "extracted_answer": extracted_answer,
            "correct": is_correct,
            "activation_a_norm": diagnostics.activation_a_norm if diagnostics else None,
            "activation_b_norm": diagnostics.activation_b_norm if diagnostics else None,
            "source_norm": diagnostics.combined_norm if diagnostics else None,
            "target_norm": diagnostics.activation_b_norm if diagnostics else None,
            "cosine_similarity": diagnostics.cosine_similarity_ab if diagnostics else None,
            "cosine_similarity_source_target": diagnostics.cosine_similarity_combined_b if diagnostics else None,
            "l2_distance_source_target": diagnostics.l2_distance_source_target if diagnostics else None,
        }
        if log_first_token_entropy:
            pred["first_token_entropy"] = first_token_entropy
        if output_confidence is not None:
            pred.update(output_confidence)
        if input_confidence is not None:
            pred.update(input_confidence)
        if benchmark in ("mmlu", "mmlu_pro", "codemmlu", "medmcqa"):
            pred["subject"] = ex.subject
            pred["choices"] = ex.choices
        elif benchmark == "math":
            pred["subject"] = ex.subject
            pred["level"] = ex.level
        elif benchmark == "rbi":
            pred["f1"] = rbi_f1
        elif benchmark == "wikimia":
            pred["wikimia_label"] = ex.label
        predictions.append(pred)

        # Print intermediate results every 10 samples
        n_total = len(predictions)
        if n_total % 10 == 0:
            n_correct = sum(1 for p in predictions if p["correct"])
            pct = 100 * n_correct / n_total
            print(
                f"  [{n_total}] running_acc="
                f"{n_correct}/{n_total} ({pct:.1f}%)",
                flush=True,
            )

    return tracker, analyzer, predictions


def save_experiment_results(
    output_dir: Path,
    config_dict: Dict,
    results: Dict,
    analyzer: ErrorAnalyzer,
    predictions: List[Dict],
    benchmark: str = "gsm8k",
):
    """Save experiment results in the same format as run_coordination.py."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    # Save results
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save error analysis
    analyzer.save(str(output_dir / f"{benchmark}_errors.json"))

    # Save predictions
    with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)


def run_single_baseline(
    model_name: str,
    short_name: str,
    examples: List,
    base_output_dir: Path,
    device: str,
    benchmark: str = "gsm8k",
    use_wandb: bool = False,
    log_first_token_entropy: bool = False,
    save_b_activations: bool = False,
    activation_layer: int = 26,
    save_output_confidence: bool = False,
    save_input_confidence: bool = False,
    input_only: bool = False,
    save_token_scores: bool = False,
) -> Dict:
    """Run a single baseline experiment using paper's methodology."""
    # Setup output directory for this experiment
    exp_name = f"baseline_{short_name.replace('.', '_').replace('-', '_')}"
    output_dir = base_output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_experiment_logging(output_dir, exp_name)

    # Initialize W&B run for this sub-experiment
    wb_run = init_wandb(
        enabled=use_wandb,
        project="llm-activations",
        name=exp_name,
        config={
            "experiment_type": "baseline",
            "model": model_name,
            "short_name": short_name,
            "n_samples": len(examples),
            "device": device,
            **GENERATION_CONFIG,
        },
        tags=["baseline", benchmark, "paper-replication"],
    )

    logger.info("=" * 60)
    logger.info("BASELINE: %s", short_name)
    logger.info("=" * 60)

    _, device_info = get_device_info()
    logger.info("Device: %s", device_info)
    logger.info("Model: %s", model_name)
    logger.info("Samples: %s", len(examples))
    logger.info("Generation config: %s", GENERATION_CONFIG)

    # Load model
    logger.info("\nLoading model...")
    config = ModelConfig(
        name=model_name,
        torch_dtype="bfloat16",
        device_map=None,
    )
    with timer(wb_run, "time/model_load_s"):
        model, tokenizer = load_model_with_info(config, device, logger)

    # Run evaluation
    logger.info("\nRunning evaluation...")
    activations_dir = None
    if save_b_activations:
        activations_dir = output_dir / "b_activations"
        activations_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving layer-%d last-token activations to: %s", activation_layer, activations_dir)

    with timer(wb_run, "time/evaluation_s"):
        _, analyzer, predictions = run_baseline_evaluation(
            model, tokenizer, examples,
            benchmark=benchmark,
            model_name=short_name,
            _logger=logger,
            wb_run=wb_run,
            log_first_token_entropy=log_first_token_entropy,
            save_b_activations=save_b_activations,
            activations_dir=activations_dir,
            activation_layer=activation_layer,
            save_output_confidence=save_output_confidence,
            save_input_confidence=save_input_confidence,
            input_only=input_only,
            save_token_scores=save_token_scores,
        )

    # Compute metrics using numerical match (not string exact match)
    n_correct = sum(1 for p in predictions if p.get("correct"))
    accuracy = n_correct / len(examples) * 100

    # Bootstrap CI for numerical accuracy
    correct_list = [1 if p.get("correct") else 0 for p in predictions]
    ci_lower, ci_upper = simple_bootstrap_ci(correct_list, n_bootstrap=1000)
    accuracy_ci = f"{accuracy:.1f}% [{ci_lower:.1f}%, {ci_upper:.1f}%]"

    results = {
        "experiment": f"Baseline {short_name}",
        "model": model_name,
        "n_samples": len(examples),
        benchmark: {
            "accuracy": accuracy,
            "accuracy_ci": accuracy_ci,
            "n_correct": n_correct,
        }
    }

    # Log final summary to W&B
    wb_run.summary_update({
        "accuracy": accuracy,
        "accuracy_ci_lower": ci_lower,
        "accuracy_ci_upper": ci_upper,
        "n_correct": n_correct,
        "n_samples": len(examples),
    })

    # Log predictions table
    wb_run.log_table(
        key="predictions",
        columns=["id", "question", "expected", "extracted_answer", "correct"],
        data=[
            [p["id"], p["question"][:200], p["expected"],
             p["extracted_answer"], p["correct"]]
            for p in predictions
        ],
    )

    logger.info("\nResults: %s", accuracy_ci)
    logger.info("Correct: %s/%s", results[benchmark]['n_correct'], len(examples))

    # Save results
    config_dict = {
        "name": exp_name,
        "description": f"Baseline evaluation of {model_name} on {benchmark.upper()} (paper replication)",
        "model": model_name,
        "generation_config": GENERATION_CONFIG,
        "evaluation": {
            "benchmark": benchmark,
            "n_samples": len(examples),
        }
    }
    save_experiment_results(output_dir, config_dict, results, analyzer, predictions, benchmark=benchmark)

    # Log result files as artifact
    wb_run.log_artifact(
        path=str(output_dir),
        name=f"{exp_name}-results",
        artifact_type="results",
        metadata={"accuracy": accuracy},
    )

    logger.info("\nResults saved to: %s", output_dir)
    wb_run.finish()

    # Clean up
    del model, tokenizer
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return results


def run_single_ac_experiment(  # pylint: disable=too-many-statements
    model_a_name: str,
    model_b_name: str,
    combination_fn_name: str,
    layer_a: int,
    layer_b: int,
    examples: List,
    base_output_dir: Path,
    device: str,
    model_cache: Dict = None,
    projection_path: str = None,
    benchmark: str = "gsm8k",
    use_wandb: bool = False,
    rand_std: float = 1.9,
    dry_run: bool = False,
    graft_position: int = -1,
    normalize_source: bool = False,
    log_first_token_entropy: bool = False,
    save_output_confidence: bool = False,
    save_input_confidence: bool = False,
    input_only: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Run a single AC experiment using paper's methodology.

    Returns:
        Tuple of (results, model_cache) - model_cache for reuse
    """
    if model_cache is None:
        model_cache = {}

    # Setup output directory for this experiment
    exp_name = f"ac_{combination_fn_name}"
    output_dir = base_output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_experiment_logging(output_dir, exp_name)

    # Initialize W&B run for this sub-experiment
    wb_run = init_wandb(
        enabled=use_wandb,
        project="llm-activations",
        name=exp_name,
        config={
            "experiment_type": "activation_communication",
            "model_a": model_a_name,
            "model_b": model_b_name,
            "layer_a": layer_a,
            "layer_b": layer_b,
            "combination_fn": combination_fn_name,
            "n_samples": len(examples),
            "device": device,
            **GENERATION_CONFIG,
        },
        tags=["ac", benchmark, "paper-replication", combination_fn_name],
    )

    logger.info("=" * 60)
    logger.info("ACTIVATION COMMUNICATION: %s", combination_fn_name)
    logger.info("=" * 60)

    _, device_info = get_device_info()
    logger.info("Device: %s", device_info)
    logger.info("Model A: %s", model_a_name)
    logger.info("Model B: %s", model_b_name)
    logger.info("Layers: %s -> %s", layer_a, layer_b)
    logger.info("Combination: %s", combination_fn_name)
    logger.info("Samples: %s", len(examples))
    logger.info("Generation config: %s", GENERATION_CONFIG)

    # Load models (with caching)
    logger.info("\nLoading models...")

    _random_source = (model_a_name == "random")
    _zero_source = (model_a_name == "zero")
    _average_act_source = (model_a_name == "average_act")
    _random_unit_source = (model_a_name == "random_unit")
    _shuffled_act_source = (model_a_name == "shuffled_act")
    _bos_token_source = (model_a_name == "bos_token")
    _prev_layer_source = (model_a_name == "prev_layer")
    _synthetic_source = (
        _random_source or _zero_source or _average_act_source
        or _random_unit_source or _shuffled_act_source
        or _bos_token_source or _prev_layer_source
    )

    with timer(wb_run, "time/model_load_s"):
        if _random_source:
            logger.info("  Model A: random Gaussian noise source (no model loaded)")
            model_a, tokenizer_a = None, None
        elif _zero_source:
            logger.info("  Model A: zero vector source (no model loaded)")
            model_a, tokenizer_a = None, None
        elif _average_act_source:
            logger.info("  Model A: average activation source (no model loaded)")
            model_a, tokenizer_a = None, None
        elif _random_unit_source:
            logger.info("  Model A: random unit vector (norm-matched) source (no model loaded)")
            model_a, tokenizer_a = None, None
        elif _shuffled_act_source:
            logger.info("  Model A: shuffled activation source (no model loaded)")
            model_a, tokenizer_a = None, None
        elif _bos_token_source:
            logger.info("  Model A: BOS token activation source (no model loaded)")
            model_a, tokenizer_a = None, None
        elif _prev_layer_source:
            logger.info("  Model A: previous layer activation source (no model loaded)")
            model_a, tokenizer_a = None, None
        elif model_a_name in model_cache:
            logger.info("  Using cached %s", model_a_name)
            model_a, tokenizer_a = model_cache[model_a_name]
        else:
            config_a = ModelConfig(name=model_a_name, torch_dtype="bfloat16", device_map=None)
            model_a, tokenizer_a = load_model_with_info(config_a, device, logger)
            model_cache[model_a_name] = (model_a, tokenizer_a)

        if model_a_name == model_b_name:
            model_b, tokenizer_b = model_a, tokenizer_a
            logger.info("  (Reusing model A for model B)")
        elif model_b_name in model_cache:
            logger.info("  Using cached %s", model_b_name)
            model_b, tokenizer_b = model_cache[model_b_name]
        else:
            config_b = ModelConfig(name=model_b_name, torch_dtype="bfloat16", device_map=None)
            model_b, tokenizer_b = load_model_with_info(config_b, device, logger)
            model_cache[model_b_name] = (model_b, tokenizer_b)

    # Get dimension info
    if _synthetic_source:
        compat = check_dimension_compatibility(model_b_name, model_b_name)
        logger.info("\nSynthetic source (%s): d_b=%s", model_a_name, compat['d_model_b'])
    else:
        compat = check_dimension_compatibility(model_a_name, model_b_name)
        logger.info(
            "\nDimension compatibility: d_a=%s, d_b=%s",
            compat['d_model_a'], compat['d_model_b'],
        )

    # Create combination function
    if _random_source:
        # std chosen so that randn(d_b)*std has the same expected norm as a typical
        # model-B hidden state: std = activation_b_norm / sqrt(d_b) ≈ 85 / sqrt(2048) ≈ 1.9
        _rand_fn = "random_add" if combination_fn_name in {"projected_sum"} else "random_noise"
        combination_fn = get_combination_function(_rand_fn, 0, compat['d_model_b'], std=rand_std)
        logger.info("Using %s combination (std=%.4f)", _rand_fn, rand_std)
    elif _zero_source:
        combination_fn = get_combination_function("zero_vector", 0, compat['d_model_b'])
        logger.info("Using zero vector combination")
    elif _average_act_source:
        combination_fn = get_combination_function("average_act", 0, compat['d_model_b'])
        logger.info("Using average activation combination")
    elif _random_unit_source:
        combination_fn = get_combination_function("random_unit", 0, compat['d_model_b'])
        logger.info("Using random unit (norm-matched) combination")
    elif _shuffled_act_source:
        combination_fn = get_combination_function("shuffled_act", 0, compat['d_model_b'])
        logger.info("Using shuffled activation combination")
    elif _bos_token_source:
        combination_fn = get_combination_function("bos_token", 0, compat['d_model_b'])
        logger.info("Using BOS token activation combination")
    elif _prev_layer_source:
        combination_fn = get_combination_function("prev_layer", 0, compat['d_model_b'])
        logger.info("Using previous layer activation combination (layer_b - 1)")
    else:
        _projection_fns = {"trained_projection", "trained", "projected_average", "projected_sum"}
        if combination_fn_name in _projection_fns and projection_path:
            combination_fn = get_combination_function(
                combination_fn_name,
                compat['d_model_a'],
                compat['d_model_b'],
                projection_path=projection_path,
                normalize_source=normalize_source,
            )
            logger.info("Loaded trained projection from: %s", projection_path)
            if normalize_source:
                logger.info("Source activations will be L2-normalized before projection")
        else:
            combination_fn = get_combination_function(
                combination_fn_name,
                compat['d_model_a'],
                compat['d_model_b'],
            )

    # Move to device if trainable
    if hasattr(combination_fn, 'to'):
        combination_fn = combination_fn.to(device)

    # Create grafting engine
    engine = ActivationGraftingEngine(
        model_a=model_a,
        model_b=model_b,
        tokenizer_a=tokenizer_a,
        tokenizer_b=tokenizer_b,
        layer_a=layer_a,
        layer_b=layer_b,
        combination_fn=combination_fn,
        extraction_point=ExtractionPoint.POST_MLP,
        grafting_mode=GraftingMode.SINGLE_SHOT,
        dry_run=dry_run,
        graft_position=graft_position,
    )

    # Run evaluation
    logger.info("\nRunning evaluation...")
    with timer(wb_run, "time/evaluation_s"):
        _, analyzer, predictions = run_grafting_evaluation(
            engine, examples,
            tokenizer_a=tokenizer_a,
            tokenizer_b=tokenizer_b,
            combination_fn_name=combination_fn_name,
            benchmark=benchmark,
            experiment_name=f"AC ({combination_fn_name})",
            _logger=logger,
            wb_run=wb_run,
            log_first_token_entropy=log_first_token_entropy,
            save_output_confidence=save_output_confidence,
            save_input_confidence=save_input_confidence,
            input_only=input_only,
        )

    # Compute metrics using numerical match (not string exact match)
    n_correct = sum(1 for p in predictions if p.get("correct"))
    accuracy = n_correct / len(examples) * 100

    # Bootstrap CI for numerical accuracy
    correct_list = [1 if p.get("correct") else 0 for p in predictions]
    ci_lower, ci_upper = simple_bootstrap_ci(correct_list, n_bootstrap=1000)
    accuracy_ci = f"{accuracy:.1f}% [{ci_lower:.1f}%, {ci_upper:.1f}%]"

    # Aggregate per-sample activation statistics (excluding NaNs)
    def _avg_metric(key):
        vals = [
            p[key] for p in predictions
            if p.get(key) is not None
            and not (isinstance(p[key], float) and np.isnan(p[key]))
        ]
        return float(np.mean(vals)) if vals else float('nan')

    avg_cosine_sim = _avg_metric("cosine_similarity_source_target")
    avg_l2_dist = _avg_metric("l2_distance_source_target")
    avg_source_norm = _avg_metric("source_norm")
    avg_target_norm = _avg_metric("target_norm")

    results = {
        "experiment": f"AC ({combination_fn_name})",
        "model_a": model_a_name,
        "model_b": model_b_name,
        "layer_a": layer_a,
        "layer_b": layer_b,
        "combination_fn": combination_fn_name,
        "n_samples": len(examples),
        "avg_cosine_similarity_source_target": avg_cosine_sim,
        "avg_l2_distance_source_target": avg_l2_dist,
        "avg_source_norm": avg_source_norm,
        "avg_target_norm": avg_target_norm,
        benchmark: {
            "accuracy": accuracy,
            "accuracy_ci": accuracy_ci,
            "n_correct": n_correct,
        }
    }

    # Log final summary to W&B
    wb_run.summary_update({
        "accuracy": accuracy,
        "accuracy_ci_lower": ci_lower,
        "accuracy_ci_upper": ci_upper,
        "n_correct": n_correct,
        "n_samples": len(examples),
        "avg_cosine_similarity_source_target": avg_cosine_sim,
        "avg_l2_distance_source_target": avg_l2_dist,
        "avg_source_norm": avg_source_norm,
        "avg_target_norm": avg_target_norm,
    })

    # Log predictions table
    wb_run.log_table(
        key="predictions",
        columns=["id", "question", "expected", "extracted_answer", "correct",
                 "activation_a_norm", "cosine_similarity"],
        data=[
            [p["id"], p["question"][:200], p["expected"],
             p["extracted_answer"], p["correct"],
             p.get("activation_a_norm"), p.get("cosine_similarity")]
            for p in predictions
        ],
    )

    logger.info("\nResults: %s", accuracy_ci)
    logger.info("Correct: %s/%s", results[benchmark]['n_correct'], len(examples))
    logger.info("Avg cosine similarity (source→target): %.4f", avg_cosine_sim)
    logger.info("Avg L2 distance (source→target): %.4f", avg_l2_dist)
    logger.info("Avg source norm: %.4f", avg_source_norm)
    logger.info("Avg target norm: %.4f", avg_target_norm)

    # Save results
    config_dict = {
        "name": exp_name,
        "description": (
            f"Activation Communication with {combination_fn_name}"
            f" on {benchmark.upper()} (paper replication)"
        ),
        "model_a": {"name": model_a_name},
        "model_b": {"name": model_b_name},
        "grafting": {
            "layer_a": layer_a,
            "layer_b": layer_b,
            "combination_fn": combination_fn_name,
            "extraction_point": "POST_MLP",
            "grafting_mode": "SINGLE_SHOT",
        },
        "generation_config": GENERATION_CONFIG,
        "evaluation": {
            "benchmark": benchmark,
            "n_samples": len(examples),
        }
    }
    save_experiment_results(output_dir, config_dict, results, analyzer, predictions, benchmark=benchmark)

    # Log result files as artifact
    wb_run.log_artifact(
        path=str(output_dir),
        name=f"{exp_name}-results",
        artifact_type="results",
        metadata={"accuracy": accuracy},
    )

    logger.info("\nResults saved to: %s", output_dir)
    wb_run.finish()

    return results, model_cache


def run_all_experiments(  # pylint: disable=too-many-statements
    n_samples: int = 100,
    seed: int = 42,
    max_new_tokens: int = 100,  # pylint: disable=unused-argument
    experiments: str = "all",
    baseline_models: str = "both",
    output_dir: Path = None,
    match_samples_from: str = None,
    use_wandb: bool = False,
    rand_std: float = 1.9,
    dry_run: bool = False,
    graft_position: int = -1,
    normalize_source: bool = False,
    benchmark: str = "gsm8k",
    mmlu_subjects: list = None,
    math_subjects: list = None,
    math_levels: list = None,
    gsm_plus_perturbation_types: list = None,
    log_first_token_entropy: bool = False,
    save_b_activations: bool = False,
    activation_layer: int = 26,
    save_output_confidence: bool = False,
    save_input_confidence: bool = False,
    input_only: bool = False,
    save_token_scores: bool = False,
    gsm8k_split: str = "test",
    medmcqa_split: str = "validation",
    mmlu_split: str = "test",
    numinamath_tir_split: str = "train",
):
    """Run all paper replication experiments sequentially."""

    device, device_info = get_device_info()

    print("\n" + "=" * 60)
    print("PAPER REPLICATION: arXiv:2501.14082")
    print("Communicating Activations Between Language Model Agents")
    print("=" * 60)
    print(f"Device: {device_info}")
    print(f"Samples: {n_samples}")
    print(f"Seed: {seed}")
    print(f"Experiments: {experiments}")
    print(f"Output: {output_dir}")
    if match_samples_from:
        print(f"Matching samples from: {match_samples_from}")

    # Load benchmark data
    print(f"\nLoading {benchmark.upper()} dataset...")

    def _load_examples(n_samples_arg):
        if benchmark == "mmlu":
            return load_mmlu_subset(n_samples=n_samples_arg, seed=seed, subjects=mmlu_subjects, split=mmlu_split)
        if benchmark == "mmlu_pro":
            return load_mmlu_pro_subset(n_samples=n_samples_arg, seed=seed, subjects=mmlu_subjects)
        if benchmark == "math":
            return load_math_subset(n_samples=n_samples_arg, seed=seed, subjects=math_subjects, levels=math_levels)
        if benchmark == "svamp":
            return load_svamp_subset(n_samples=n_samples_arg, seed=seed)
        if benchmark == "gsm_plus":
            return load_gsm_plus_subset(
                n_samples=n_samples_arg, seed=seed,
                perturbation_types=gsm_plus_perturbation_types,
            )
        if benchmark == "gsm_symbolic":
            return load_gsm_symbolic_subset(n_samples=n_samples_arg, seed=seed)
        if benchmark == "codemmlu":
            return load_codemmlu_subset(n_samples=n_samples_arg, seed=seed)
        if benchmark == "rbi":
            return load_rbi_subset(n_samples=n_samples_arg, seed=seed)
        if benchmark == "medmcqa":
            return load_medmcqa_subset(n_samples=n_samples_arg, seed=seed, split=medmcqa_split)
        if benchmark == "wikimia":
            return load_wikimia_subset(n_samples=n_samples_arg, seed=seed)
        if benchmark == "numinamath_tir":
            return load_numinamath_tir_subset(n_samples=n_samples_arg, seed=seed, split=numinamath_tir_split)
        return load_gsm8k_subset(n_samples=n_samples_arg, seed=seed, split=gsm8k_split)

    if match_samples_from:
        # Load sample IDs from previous experiment
        sample_ids = load_sample_ids_from_predictions(match_samples_from)
        print(f"Loaded {len(sample_ids)} sample IDs from previous experiment")

        all_examples = _load_examples(None)
        print(f"Loaded full {benchmark.upper()} test set: {len(all_examples)} examples")
        examples = filter_examples_by_ids(all_examples, sample_ids)
        print(f"Matched {len(examples)}/{len(sample_ids)} samples")

        if len(examples) != len(sample_ids):
            n_missing = len(sample_ids) - len(examples)
            print(
                f"WARNING: Could not find all samples. "
                f"Missing {n_missing} samples."
            )
    else:
        examples = _load_examples(n_samples)
        print(f"Loaded {len(examples)} examples")

    all_results = []

    # Run baselines
    if experiments in ["baselines", "all"]:
        print("\n" + "=" * 60)
        print("RUNNING BASELINES")
        print("=" * 60)

        # Paper uses instruct-tuned models for the baselines
        # Use local paths if available (same as MODEL_A_NAME and MODEL_B_NAME)
        all_baselines = [
            (MODEL_A_NAME, os.path.basename(MODEL_A_NAME.rstrip("/"))),
            (MODEL_B_NAME, os.path.basename(MODEL_B_NAME.rstrip("/"))),
        ]
        if baseline_models == "a":
            baselines = all_baselines[:1]
        elif baseline_models == "b":
            baselines = all_baselines[1:]
        else:
            baselines = all_baselines

        for model_name, short_name in baselines:
            result = run_single_baseline(
                model_name=model_name,
                short_name=short_name,
                examples=examples,
                base_output_dir=output_dir,
                device=device,
                benchmark=benchmark,
                use_wandb=use_wandb,
                log_first_token_entropy=log_first_token_entropy,
                save_b_activations=save_b_activations,
                activation_layer=activation_layer,
                save_output_confidence=save_output_confidence,
                save_input_confidence=save_input_confidence,
                input_only=input_only,
                save_token_scores=save_token_scores,
            )
            all_results.append(result)

    # Run AC experiments
    if experiments in ["ac", "all"]:
        print("\n" + "=" * 60)
        print("RUNNING ACTIVATION COMMUNICATION")
        print("=" * 60)

        # Paper uses instruct-tuned models for AC experiments
        # Use configuration from PAPER_CONFIG at the top of the file
        model_a_name = MODEL_A_NAME
        model_b_name = MODEL_B_NAME
        layer_a = PAPER_CONFIG["layer_a"]
        layer_b = PAPER_CONFIG["layer_b"]

        # Support both single string and list of combination functions
        comb_fn_config = PAPER_CONFIG["combination_fn"]
        combination_fns = comb_fn_config if isinstance(comb_fn_config, list) else [comb_fn_config]

        projection_path = PAPER_CONFIG["projection_path"]
        model_cache = {}

        for comb_fn in combination_fns:
            result, model_cache = run_single_ac_experiment(
                model_a_name=model_a_name,
                model_b_name=model_b_name,
                combination_fn_name=comb_fn,
                layer_a=layer_a,
                layer_b=layer_b,
                examples=examples,
                base_output_dir=output_dir,
                device=device,
                model_cache=model_cache,
                projection_path=projection_path if comb_fn in {"trained_projection", "trained", "projected_average", "projected_sum"} else None,
                benchmark=benchmark,
                use_wandb=use_wandb,
                rand_std=rand_std,
                dry_run=dry_run,
                graft_position=graft_position,
                normalize_source=normalize_source,
                log_first_token_entropy=log_first_token_entropy,
                save_output_confidence=save_output_confidence,
                save_input_confidence=save_input_confidence,
                input_only=input_only,
            )
            all_results.append(result)

        # Clean up model cache
        del model_cache
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30} {'Accuracy':>15}")
    print("-" * 50)

    for r in all_results:
        name = r.get("experiment", "Unknown")
        acc_ci = r.get(benchmark, {}).get("accuracy_ci", "N/A")
        print(f"{name:<30} {acc_ci:>15}")
        cos_sim = r.get("avg_cosine_similarity_source_target")
        l2_dist = r.get("avg_l2_distance_source_target")
        src_norm = r.get("avg_source_norm")
        tgt_norm = r.get("avg_target_norm")
        if cos_sim is not None:
            print(f"  Avg cosine similarity (source→target): {cos_sim:.4f}")
            print(f"  Avg L2 distance       (source→target): {l2_dist:.4f}")
            print(f"  Avg source norm:                       {src_norm:.4f}")
            print(f"  Avg target norm:                       {tgt_norm:.4f}")

    # Save overall summary
    summary_file = output_dir / "overall_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "device": device_info,
            "n_samples": n_samples,
            "seed": seed,
            "experiments": experiments,
            "results": all_results,
        }, f, indent=2)

    print(f"\nOverall summary saved to: {summary_file}")
    print(f"Individual experiment results in: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Paper replication experiments for arXiv:2501.14082",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Arguments reference
───────────────────
  --experiments     Which experiment type to run.
                      baselines — run each model standalone, no grafting
                      ac        — run activation communication (grafting from A into B)
                      all       — run both baselines and ac
                    Default: all
                    Example: --experiments ac

  --benchmark       Benchmark to evaluate on.
                      gsm8k        — math word problems; numeric match correctness
                      svamp        — math word problems (challenge set); numeric match
                      gsm_plus     — harder GSM8K variants; numeric match
                      gsm_symbolic — GSM8K with symbolic variable substitution; numeric match
                      math         — competition math; symbolic equivalence (math_equal)
                      mmlu         — multiple-choice A-D, general knowledge; exact letter match
                      mmlu_pro     — multiple-choice A-J, harder MMLU; exact letter match
                      codemmlu     — multiple-choice A-D, coding knowledge; exact letter match
                      rbi          — open-ended regulatory QA; token F1 >= 0.3
                      medmcqa      — medical MCQ A-D, single-choice only; exact letter match
                    Default: gsm8k
                    Example: --benchmark gsm8k

  --model-a         Path to source model checkpoint (model A, the sender), OR one of
                    these special synthetic vectors (no model loaded, no projection needed):
                      zero                  — zero vector; replaces last-token activation with zeros
                      random                — Gaussian noise (std=--rand-std); random direction + magnitude
                      random_unit           — Gaussian noise normalized to ||b||; random direction, real magnitude
                      average_act           — mean of all other prompt token activations at layer_b
                      shuffled_act          — b with hidden dim randomly permuted; preserves norm, destroys structure
                      bos_token             — BOS token's hidden state at layer_b
                      prev_layer — last-token hidden state from layer_b - 1
                    When using a real model path, --projection-path is also required.
                    Example: --model-a $CKPTS/bounhar3B
                    Example: --model-a zero

  --model-b         Path to target model checkpoint (model B, the receiver).
                    Default: from PAPER_CONFIG
                    Example: --model-b $CKPTS/Qwen2.5-3B-Instruct

  --layer-a         Layer index to extract the activation from in model A.
                    Default: from PAPER_CONFIG (26 for Qwen layer-26 experiments)
                    Example: --layer-a 26

  --layer-b         Layer index to inject the activation into in model B.
                    Default: from PAPER_CONFIG (26 for Qwen layer-26 experiments)
                    Example: --layer-b 26

  --projection-path Path to a trained W projection matrix (.pt file) mapping
                    model A's hidden space to model B's hidden space.
                    Not needed for synthetic sources (zero, random, average_act, etc.).
                    Default: auto-resolved from projections_26-26/mapping.json
                    Example: --projection-path projections_26-26/w_bounhar3b_to_qwen3b.pt

  --baseline-models Which models to evaluate in standalone (no-graft) baseline mode.
                      a    — model A only
                      b    — model B only
                      both — both models (default)
                    Only used when --experiments baselines or all.
                    Example: --baseline-models b

  --combination-fn  Per-source projection function applied before grafting.
                      trained_projection — W @ a  (standard, replaces b entirely)
                      projected_average  — blend of W@a and b with alpha=0.5
                    Default: trained_projection
                    Example: --combination-fn trained_projection

  --graft-position  Token position in model B's prompt where grafting is applied.
                      last        — final prompt token (default; the generation pivot)
                      first       — first token (BOS)
                      penultimate — second-to-last token
                    Example: --graft-position last

  --normalize-source  Flag. L2-normalize source activations before applying W.
                    Use this when W was trained with --loss-fn normalized_mse.
                    Default: off

  --rand-std        Standard deviation for the random Gaussian noise source.
                    Only relevant with --model-a random.
                    Default: 1.9  (calibrated so ||randn(d)*std|| ≈ typical ||b||)
                    Example: --rand-std 1.9

  --n-samples       Number of benchmark examples to evaluate.
                    Default: 100
                    Example: --n-samples 1000

  --seed            Random seed for data sampling and generation.
                    Default: 42
                    Example: --seed 42

  --max-new-tokens  Maximum number of new tokens to generate per example.
                    Default: 100 (override via GENERATION_CONFIG in the script; paper uses 2048)
                    Example: --max-new-tokens 2048

  --output-dir      Parent directory for results. The actual subfolder is auto-generated as:
                      {benchmark}_{experiments}_{model-a}_{graft-position}_L{layer-b}_{model-b}
                    For example: results_greedy_decoding/gsm8k_ac_random_last_L26_Qwen2.5-3B-Instruct
                    Inside that subfolder:
                      config.yaml              — full run configuration
                      predictions.json         — per-example prompt/output/correctness
                      results.json             — aggregate accuracy and metadata
                      <benchmark>_errors.json  — incorrect predictions only
                      experiment.log           — full log
                    Default: results/paper_replication_<timestamp>/<auto-name>
                    Example: --output-dir results_greedy_decoding

  --match-samples-from  Path to a predictions.json from a previous run. Re-uses
                    the exact same sample indices for a fair comparison.
                    Example: --match-samples-from results_no_truncation/gsm8k_baseline_qwen3B_1000/predictions.json

  --mmlu-subjects   Filter MMLU or MMLU-Pro to specific subjects (default: all).
                    Example: --mmlu-subjects abstract_algebra college_math

  --math-subjects   Filter MATH to specific subjects (default: all 7).
                    Choices: Algebra Counting_and_Probability Geometry
                             Number_Theory Prealgebra Precalculus Intermediate_Algebra
                    Example: --math-subjects Algebra Geometry

  --math-levels     Filter MATH to specific difficulty levels 1-5 (default: all).
                    Example: --math-levels 4 5

  --gsm-plus-perturbation-types  Filter GSM-Plus to specific perturbation types (default: all).
                    Example: --gsm-plus-perturbation-types 'numerical substitution' 'digit expansion'

  --wandb           Flag. Enable Weights & Biases experiment logging.

Example commands
────────────────
  # Baseline: target model only
  # → saves to results_no_truncation/gsm8k_baselines_Qwen2.5-3B-Instruct_last_L26_none/
  python experiments/run_paper_replication.py \\
      --experiments baselines --model-a $CKPTS/Qwen2.5-3B-Instruct \\
      --baseline-models a --benchmark gsm8k --n-samples 1000 --seed 42 \\
      --output-dir results_no_truncation

  # AC with a real fine-tuned source model
  # → saves to results_no_truncation/gsm8k_ac_bounhar3B_last_L26_Qwen2.5-3B-Instruct/
  python experiments/run_paper_replication.py \\
      --experiments ac --model-a $CKPTS/bounhar3B \\
      --model-b $CKPTS/Qwen2.5-3B-Instruct \\
      --layer-a 26 --layer-b 26 --benchmark gsm8k --n-samples 1000 --seed 42 \\
      --projection-path projections_26-26/w_bounhar3b_to_qwen3b.pt \\
      --output-dir results_no_truncation

  # AC with zero vector (no model A needed, no projection needed)
  # → saves to results_greedy_decoding/gsm8k_ac_zero_last_L26_Qwen2.5-3B-Instruct/
  python experiments/run_paper_replication.py \\
      --experiments ac --model-a zero \\
      --model-b $CKPTS/Qwen2.5-3B-Instruct \\
      --benchmark gsm8k --n-samples 1000 --seed 42 \\
      --output-dir results_greedy_decoding

  # AC with norm-matched random vector, different layer and position
  # → saves to results_greedy_decoding/gsm8k_ac_random_unit_penultimate_L20_Qwen2.5-3B-Instruct/
  python experiments/run_paper_replication.py \\
      --experiments ac --model-a random_unit \\
      --model-b $CKPTS/Qwen2.5-3B-Instruct \\
      --layer-b 20 --graft-position penultimate \\
      --benchmark gsm8k --n-samples 1000 --seed 42 \\
      --output-dir results_greedy_decoding
"""
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        choices=["baselines", "ac", "all"],
        help="Which experiments to run"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of GSM8K samples (paper uses 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--match-samples-from",
        type=str,
        default=None,
        help="Path to predictions.json from a previous experiment to use the same samples"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--model-a",
        type=str,
        default=None,
        help="Override model A path (default: from TRANSFORMERS_CACHE config)"
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default=None,
        help="Override model B path (default: from TRANSFORMERS_CACHE config)"
    )
    parser.add_argument(
        "--layer-a",
        type=int,
        default=None,
        help="Override layer to extract from model A (default: from PAPER_CONFIG)"
    )
    parser.add_argument(
        "--layer-b",
        type=int,
        default=None,
        help="Override layer to inject into model B (default: from PAPER_CONFIG)"
    )
    parser.add_argument(
        "--projection-path",
        type=str,
        default=None,
        help="Override projection matrix path (default: from PAPER_CONFIG)"
    )
    parser.add_argument(
        "--baseline-models",
        type=str,
        default="both",
        choices=["a", "b", "both"],
        help="Which models to evaluate in baseline: 'a' (source only), 'b' (target only), 'both' (default)"
    )
    parser.add_argument(
        "--combination-fn",
        type=str,
        default=None,
        help="Override combination function (default: from PAPER_CONFIG, e.g. trained_projection, projected_average)"
    )
    parser.add_argument(
        "--rand-std",
        type=float,
        default=1.9,
        help="Standard deviation for random Gaussian noise source (used only when --model-a random, default: 1.9)"
    )
    parser.add_argument(
        "--graft-position",
        type=str,
        default="last",
        choices=["last", "first", "penultimate"],
        help="Token position in prompt_b to graft the source activation at: "
             "'last' (default, final prompt token), 'first' (first token), "
             "'penultimate' (second-to-last token)",
    )
    parser.add_argument(
        "--normalize-source",
        action="store_true",
        default=False,
        help="L2-normalize source activations before applying projection W. "
             "Required when W was trained with --loss-fn normalized_mse.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "mmlu", "mmlu_pro", "math", "svamp", "gsm_plus", "gsm_symbolic", "codemmlu", "rbi", "medmcqa", "wikimia", "numinamath_tir"],
        help="Benchmark to evaluate on (default: gsm8k)",
    )
    parser.add_argument(
        "--gsm8k-split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="GSM8K split to use when --benchmark gsm8k (default: test). Use 'train' for memorization detection experiments.",
    )
    parser.add_argument(
        "--medmcqa-split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="MedMCQA split to use when --benchmark medmcqa (default: validation). Use 'train' for memorization detection experiments.",
    )
    parser.add_argument(
        "--mmlu-split",
        type=str,
        default="test",
        choices=["test", "validation", "dev", "auxiliary_train"],
        help="MMLU split to use when --benchmark mmlu (default: test). Use 'auxiliary_train' for memorization detection experiments.",
    )
    parser.add_argument(
        "--numinamath-tir-split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="NuminaMath-TIR split to use when --benchmark numinamath_tir (default: train). Test has ~99 examples.",
    )
    parser.add_argument(
        "--mmlu-subjects",
        type=str,
        nargs="+",
        default=None,
        help="MMLU/MMLU-Pro subjects/categories to include (default: all). E.g. --mmlu-subjects abstract_algebra math",
    )
    parser.add_argument(
        "--math-subjects",
        type=str,
        nargs="+",
        default=None,
        help="MATH subjects to include (default: all 7). E.g. --math-subjects Algebra Geometry",
    )
    parser.add_argument(
        "--math-levels",
        type=int,
        nargs="+",
        default=None,
        help="MATH difficulty levels to include, 1-5 (default: all). E.g. --math-levels 4 5",
    )
    parser.add_argument(
        "--gsm-plus-perturbation-types",
        type=str,
        nargs="+",
        default=None,
        help=(
            "GSM-Plus perturbation types to include (default: all). "
            "E.g. --gsm-plus-perturbation-types 'numerical substitution' 'digit expansion'"
        ),
    )
    parser.add_argument(
        "--save-b-activations",
        action="store_true",
        default=False,
        help=(
            "If set, save model B's last-token hidden state at --activation-layer for every "
            "example in a 'b_activations/' subfolder of the output directory. Each file is named "
            "<example_id>.pt and contains a 1-D CPU tensor of shape (hidden_size,). "
            "Used to collect training data for the cross-task routing probe (Option A). "
            "Only applies to baseline evaluations (no grafting)."
        ),
    )
    parser.add_argument(
        "--activation-layer",
        type=int,
        default=26,
        help="Transformer layer index from which to capture model B's activation when "
             "--save-b-activations is set. Default: 26 (standard graft layer).",
    )
    parser.add_argument(
        "--log-first-output-token-entropy",
        action="store_true",
        default=False,
        help=(
            "If set, compute and log the entropy of the first generated token's logit distribution "
            "to predictions.json as 'first_token_entropy'. For baselines: uses output_scores from "
            "model.generate(). For AC: uses a separate model_b forward pass on the prompt. "
            "Disabled by default (no overhead when not set)."
        ),
    )
    parser.add_argument(
        "--save-output-confidence",
        action="store_true",
        default=False,
        help=(
            "If set, compute and log three per-token confidence metrics over the full generated "
            "sequence to predictions.json: "
            "'mean_token_confidence' (mean softmax prob of chosen token), "
            "'mean_token_log_prob' (mean log prob of chosen token), "
            "'seq_log_prob' (sum of log probs = sequence log-likelihood). "
            "Uses the actually sampled token at each step, so under greedy decoding this equals "
            "the top-1 probability. Requires output_scores=True in generate(), adding minor overhead."
        ),
    )
    parser.add_argument(
        "--save-input-confidence",
        action="store_true",
        default=False,
        help=(
            "If set, compute and log three per-token confidence metrics over the input prompt "
            "to predictions.json (prompt perplexity in probability space): "
            "'input_mean_token_confidence' (mean P(token_i | prefix)), "
            "'input_mean_token_log_prob' (mean log P(token_i | prefix)), "
            "'input_seq_log_prob' (sum of log probs over prompt tokens). "
            "Requires an extra forward pass on the prompt — adds overhead proportional to prompt length."
        ),
    )
    parser.add_argument(
        "--input-only",
        action="store_true",
        default=False,
        help=(
            "If set, skip autoregressive generation entirely and only run a single forward pass "
            "on the prompt to collect input confidence metrics. Implies --save-input-confidence. "
            "predictions.json will have generated='' and correct=null. "
            "Much faster than full generation — useful for prompt perplexity sweeps."
        ),
    )
    parser.add_argument(
        "--save-token-scores",
        action="store_true",
        default=False,
        help=(
            "If set (together with --input-only), save per-token log-probabilities and "
            "Min-K++ z-scores to predictions.json as 'token_log_probs' and "
            "'token_minkpp_scores' (lists of floats, one per prompt+GT token). "
            "Required for post-hoc computation of Loss, Zlib, Min-K%%, and Min-K++ baselines "
            "via experiments/min_plus_plus.py. Adds minor overhead (one extra softmax over "
            "the full vocabulary per token position)."
        ),
    )
    parser.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        default=None,
        help=(
            "Enable stochastic sampling during generation (sets do_sample=True in generation config). "
            "When set, --temperature and --top-p control the distribution. "
            "Default: greedy decoding (do_sample=False). "
            "Use for self-consistency experiments: run multiple times with different --seed values "
            "to get diverse samples from the same examples."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature (only used when --do-sample is set). "
            "Higher values increase diversity; lower values concentrate probability. "
            "Default: 0.7 (from GENERATION_CONFIG). Typical range: 0.5–1.0."
        ),
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=None,
        help=(
            "Nucleus sampling threshold (only used when --do-sample is set). "
            "Keeps the smallest set of tokens whose cumulative probability exceeds top_p. "
            "Default: 0.9 (from GENERATION_CONFIG)."
        ),
    )

    args = parser.parse_args()
    set_seed(args.seed)

    # Apply sampling overrides
    if args.do_sample is not None:
        GENERATION_CONFIG["do_sample"] = args.do_sample
    if args.temperature is not None:
        GENERATION_CONFIG["temperature"] = args.temperature
    if args.top_p is not None:
        GENERATION_CONFIG["top_p"] = args.top_p

    # If --model-a is not provided for an AC experiment, run AC with a random
    # noise source but skip the write-back (dry_run=True).  Model B is
    # unmodified, but the hook still fires and logs cosine similarities.
    _dry_run = False
    if args.model_a is None and args.experiments in ["ac", "all"]:
        print(
            "[INFO] --model-a not provided: AC experiment will use random noise "
            "source with no write-back (dry_run=True)"
        )
        args.model_a = "random"
        _dry_run = True

    # Apply CLI overrides to globals
    if args.model_a:
        global MODEL_A_NAME  # pylint: disable=global-statement
        MODEL_A_NAME = args.model_a
    if args.model_b:
        global MODEL_B_NAME  # pylint: disable=global-statement
        MODEL_B_NAME = args.model_b
    if args.layer_a is not None:
        PAPER_CONFIG["layer_a"] = args.layer_a
    if args.layer_b is not None:
        PAPER_CONFIG["layer_b"] = args.layer_b
    if args.projection_path:
        PAPER_CONFIG["projection_path"] = args.projection_path
    if args.combination_fn:
        PAPER_CONFIG["combination_fn"] = args.combination_fn

    # Setup output directory
    # --output-dir is treated as a parent folder; the actual results subfolder
    # is auto-generated as {benchmark}_{experiments}_{model-a}_{graft-position}_{layer-b}_{model-b}
    def _short_name(path: str) -> str:
        """Return the last path component, or the special value as-is."""
        return os.path.basename(path.rstrip("/\\")) if os.sep in path or "/" in path else path

    model_a_slug = _short_name(MODEL_A_NAME) if MODEL_A_NAME else "none"
    model_b_slug = _short_name(MODEL_B_NAME) if MODEL_B_NAME else "none"
    layer_b_val = PAPER_CONFIG.get("layer_b", 26)
    decoding_tag = "greedy" if not GENERATION_CONFIG["do_sample"] else "sampling"
    seed_tag = f"_seed{args.seed}" if GENERATION_CONFIG["do_sample"] else ""
    if args.experiments == "baselines":
        if args.baseline_models == "a":
            auto_name = f"{args.benchmark}_baseline_{model_a_slug}_{decoding_tag}{seed_tag}_{args.n_samples}"
        elif args.baseline_models == "b":
            auto_name = f"{args.benchmark}_baseline_{model_b_slug}_{decoding_tag}{seed_tag}_{args.n_samples}"
        else:
            auto_name = f"{args.benchmark}_baselines_{model_a_slug}_{model_b_slug}_{decoding_tag}{seed_tag}_{args.n_samples}"
    else:
        auto_name = f"{args.benchmark}_{args.experiments}_{model_a_slug}_{args.graft_position}_L{layer_b_val}_{model_b_slug}_{decoding_tag}{seed_tag}_{args.n_samples}"

    if args.output_dir:
        output_dir = Path(args.output_dir) / auto_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"paper_replication_{timestamp}" / auto_name

    output_dir.mkdir(parents=True, exist_ok=True)

    _graft_position = {"last": -1, "first": 0, "penultimate": -2}[args.graft_position]

    run_all_experiments(
        n_samples=args.n_samples,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        experiments=args.experiments,
        baseline_models=args.baseline_models,
        output_dir=output_dir,
        match_samples_from=args.match_samples_from,
        use_wandb=args.wandb,
        rand_std=args.rand_std,
        dry_run=_dry_run,
        graft_position=_graft_position,
        normalize_source=args.normalize_source,
        benchmark=args.benchmark,
        mmlu_subjects=args.mmlu_subjects,
        math_subjects=args.math_subjects,
        math_levels=args.math_levels,
        gsm_plus_perturbation_types=args.gsm_plus_perturbation_types,
        log_first_token_entropy=args.log_first_output_token_entropy,
        save_b_activations=args.save_b_activations,
        activation_layer=args.activation_layer,
        save_output_confidence=args.save_output_confidence,
        save_input_confidence=args.save_input_confidence,
        input_only=args.input_only,
        save_token_scores=args.save_token_scores,
        gsm8k_split=args.gsm8k_split,
        medmcqa_split=args.medmcqa_split,
        mmlu_split=args.mmlu_split,
        numinamath_tir_split=args.numinamath_tir_split,
    )


# =============================================================================
# Modal GPU Execution
# =============================================================================

MODAL_PROJECTION_PATH = "/root/projections_26-26/w_3b_to_8b.pt"
# MODEL_A_NAME and MODEL_B_NAME are now defined at the top of the file (lines 79-104)


# @app.function(
#     image=modal_image,
#     gpu="A100",
#     timeout=7200,
#     secrets=[
#         modal.Secret.from_name("huggingface-secret"),
#         modal.Secret.from_dotenv(),
#     ],
# )
def evaluate_gsm8k(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    mode: str,
    examples_json: List[Dict[str, Any]],
    use_wandb: bool,
) -> Dict[str, Any]:
    """Run a single GSM8K experiment on a Modal GPU.

    Args:
        mode: One of "baseline_3b", "baseline_8b", "ac_trained_projection".
        examples_json: Pre-loaded GSM8K examples as dicts (loaded once locally,
            passed to every container to guarantee identical samples).
        use_wandb: Whether to log to Weights & Biases.

    Returns:
        Dict with experiment name, accuracy, CI, and predictions list.
    """
    # -- container-side imports ------------------------------------------------
    import sys as _sys  # pylint: disable=import-outside-toplevel
    _sys.path.insert(0, "/root")

    import time as _time  # pylint: disable=import-outside-toplevel
    import torch as _torch  # pylint: disable=import-outside-toplevel
    from tqdm import tqdm as _tqdm  # pylint: disable=import-outside-toplevel

    from src.benchmarks.reasoning_tasks import (  # pylint: disable=import-outside-toplevel
        ReasoningExample as _RE,
    )
    from src.communication.activation_graft import (  # pylint: disable=import-outside-toplevel
        ActivationGraftingEngine as _Engine,
        GraftingMode as _GraftMode,
    )
    from src.communication.combination_functions import (  # pylint: disable=import-outside-toplevel
        get_combination_function as _get_comb_fn,
    )
    from src.evaluation.metrics import (  # pylint: disable=import-outside-toplevel
        numerical_match as _num_match,
    )
    from src.models.activation_extractor import (  # pylint: disable=import-outside-toplevel
        ExtractionPoint as _ExtPt,
    )
    from src.models.model_loader import (  # pylint: disable=import-outside-toplevel
        load_model_and_tokenizer as _load_model,
    )
    from src.models.model_registry import (  # pylint: disable=import-outside-toplevel
        check_dimension_compatibility as _check_compat,
    )
    from src.utils.config import (  # pylint: disable=import-outside-toplevel
        ModelConfig as _MCfg,
        set_seed as _set_seed,
    )
    from src.utils.wandb_utils import (  # pylint: disable=import-outside-toplevel
        init_wandb as _init_wandb,
    )

    # -- setup -----------------------------------------------------------------
    _set_seed(42)
    print(f"[{mode}] GPU: {_torch.cuda.get_device_name(0)}")

    # Reconstruct ReasoningExample objects from the JSON dicts passed in
    examples = [_RE(**d) for d in examples_json]
    print(f"[{mode}] Using {len(examples)} GSM8K examples (pre-loaded)")

    device = "cuda"

    # Determine experiment metadata
    if mode == "baseline_3b":
        exp_name, short = "baseline_3_2_3B", "3.2-3B"
        tags = ["baseline", "gsm8k", "paper-replication", "3B"]
    elif mode == "baseline_8b":
        exp_name, short = "baseline_3_1_8B", "3.1-8B"
        tags = ["baseline", "gsm8k", "paper-replication", "8B"]
    elif mode == "ac_trained_projection":
        exp_name, short = "ac_trained_projection", "AC(W)"
        tags = ["ac", "gsm8k", "paper-replication", "trained_projection"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    wb = _init_wandb(
        enabled=use_wandb,
        project="llm-activations",
        name=exp_name,
        config={"mode": mode, "n_samples": len(examples),
                **GENERATION_CONFIG},
        tags=tags,
    )

    # -- model loading ---------------------------------------------------------
    predictions: List[Dict] = []

    if mode in ("baseline_3b", "baseline_8b"):
        model_name = MODEL_A_NAME if mode == "baseline_3b" else MODEL_B_NAME
        cfg = _MCfg(name=model_name, torch_dtype="bfloat16", device_map=None)
        model, tokenizer = _load_model(cfg, device)
        print(f"[{mode}] Loaded {model_name}")

        for idx, ex in enumerate(_tqdm(examples, desc=f"Evaluating {short}")):
            prompt = format_chat_prompt(ex.question, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            t0 = _time.time()
            with _torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                    do_sample=GENERATION_CONFIG["do_sample"],
                    top_p=GENERATION_CONFIG["top_p"],
                    temperature=GENERATION_CONFIG["temperature"],
                    pad_token_id=tokenizer.pad_token_id,
                )
            elapsed = _time.time() - t0

            gen = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            ans = extract_answer_from_response(gen)
            ok = _num_match(ans or "", ex.answer) if ans else False

            n_ok = sum(1 for p in predictions if p["correct"]) + int(ok)
            wb.log({
                "sample/correct": int(ok),
                "sample/running_accuracy": n_ok / (idx + 1),
                "sample/generation_time_s": elapsed,
            }, step=idx)

            predictions.append({
                "id": ex.id, "question": ex.question, "expected": ex.answer,
                "generated": gen, "extracted_answer": ans, "correct": ok,
            })

        experiment_label = f"Baseline {short}"

    else:
        # AC trained projection – load both models
        cfg_a = _MCfg(name=MODEL_A_NAME, torch_dtype="bfloat16", device_map=None)
        model_a, tok_a = _load_model(cfg_a, device)
        print(f"[{mode}] Model A loaded")

        cfg_b = _MCfg(name=MODEL_B_NAME, torch_dtype="bfloat16", device_map=None)
        model_b, tok_b = _load_model(cfg_b, device)
        print(f"[{mode}] Model B loaded")

        compat = _check_compat(MODEL_A_NAME, MODEL_B_NAME)
        layer_a = layer_b = 26

        comb_fn = _get_comb_fn(
            "trained_projection",
            compat["d_model_a"], compat["d_model_b"],
            projection_path=MODAL_PROJECTION_PATH,
        )
        comb_fn = comb_fn.to(device)

        engine = _Engine(
            model_a=model_a, model_b=model_b,
            tokenizer_a=tok_a, tokenizer_b=tok_b,
            layer_a=layer_a, layer_b=layer_b,
            combination_fn=comb_fn,
            extraction_point=_ExtPt.POST_MLP,
            grafting_mode=_GraftMode.SINGLE_SHOT,
        )

        for idx, ex in enumerate(_tqdm(examples, desc="Evaluating AC(W)")):
            prompt_a = format_chat_prompt(ex.question, tok_a)
            prompt_b = format_chat_prompt(ex.question, tok_b)

            t0 = _time.time()
            output, diag = engine.generate(
                prompt_a=prompt_a, prompt_b=prompt_b,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                return_diagnostics=True,
                do_sample=GENERATION_CONFIG["do_sample"],
                top_p=GENERATION_CONFIG["top_p"],
                temperature=GENERATION_CONFIG["temperature"],
            )
            elapsed = _time.time() - t0
            output = output.strip()

            ans = extract_answer_from_response(output)
            ok = _num_match(ans or "", ex.answer) if ans else False

            n_ok = sum(1 for p in predictions if p["correct"]) + int(ok)
            wb.log({
                "sample/correct": int(ok),
                "sample/running_accuracy": n_ok / (idx + 1),
                "sample/generation_time_s": elapsed,
                "activation/a_norm": diag.activation_a_norm,
                "activation/b_norm": diag.activation_b_norm,
                "activation/combined_norm": diag.combined_norm,
                "activation/cosine_sim_ab": diag.cosine_similarity_ab,
                "activation/output_entropy": diag.output_entropy,
            }, step=idx)

            predictions.append({
                "id": ex.id, "question": ex.question, "expected": ex.answer,
                "generated": output, "extracted_answer": ans, "correct": ok,
                "activation_a_norm": diag.activation_a_norm,
                "activation_b_norm": diag.activation_b_norm,
                "cosine_similarity": diag.cosine_similarity_ab,
            })

            if len(predictions) % 10 == 0:
                n_c = sum(1 for p in predictions if p["correct"])
                print(f"  [{len(predictions)}] acc={n_c}/{len(predictions)}"
                      f" ({100*n_c/len(predictions):.1f}%)")

        experiment_label = "AC (trained_projection)"

    # -- aggregate results -----------------------------------------------------
    n_correct = sum(1 for p in predictions if p["correct"])
    accuracy = n_correct / len(examples) * 100
    correct_list = [1 if p["correct"] else 0 for p in predictions]
    ci_lower, ci_upper = simple_bootstrap_ci(correct_list)
    accuracy_ci = f"{accuracy:.1f}% [{ci_lower:.1f}%, {ci_upper:.1f}%]"

    wb.summary_update({
        "accuracy": accuracy,
        "accuracy_ci_lower": ci_lower,
        "accuracy_ci_upper": ci_upper,
        "n_correct": n_correct,
        "n_samples": len(examples),
    })
    wb.log_table(
        key="predictions",
        columns=["id", "question", "expected", "extracted_answer", "correct"],
        data=[[p["id"], p["question"][:200], p["expected"],
               p["extracted_answer"], p["correct"]] for p in predictions],
    )
    wb.finish()

    print(f"[{mode}] Done. {accuracy_ci}")
    return {
        "experiment": experiment_label,
        "mode": mode,
        "n_samples": len(examples),
        "accuracy": accuracy,
        "accuracy_ci": accuracy_ci,
        "n_correct": n_correct,
        "predictions": predictions,
    }


# @app.local_entrypoint()
def modal_main(
    experiments: str = "all",
    n_samples: int = 100,
    seed: int = 42,
    wandb: bool = False,
    output_dir: str = "",
):
    """Run paper replication experiments in parallel on Modal GPUs."""
    modes: List[str] = []
    if experiments in ("baselines", "all"):
        modes.extend(["baseline_3b", "baseline_8b"])
    if experiments in ("ac", "all"):
        modes.append("ac_trained_projection")

    # Load GSM8K samples ONCE locally so every container gets identical data
    examples = load_gsm8k_subset(n_samples=n_samples, seed=seed)
    examples_json = [
        {"id": ex.id, "question": ex.question, "answer": ex.answer,
         "chain_of_thought": ex.chain_of_thought,
         "numerical_answer": ex.numerical_answer}
        for ex in examples
    ]

    print("=" * 60)
    print("PAPER REPLICATION (Modal): arXiv:2501.14082")
    print("=" * 60)
    print(f"Modes:   {modes}")
    print(f"Samples: {len(examples_json)}  Seed: {seed}  W&B: {wandb}")
    print(f"Sample IDs: {[e['id'] for e in examples_json[:5]]}...")
    print()

    # Launch all experiments in parallel on separate Modal GPUs
    args = [(mode, examples_json, wandb) for mode in modes]
    all_results = list(evaluate_gsm8k.starmap(args))

    # Setup output directory
    if output_dir:
        out = Path(output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("results") / f"paper_replication_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    # Save per-mode results
    for result in all_results:
        mode_dir = out / result["mode"]
        mode_dir.mkdir(parents=True, exist_ok=True)
        preds = result.pop("predictions")
        with open(mode_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        with open(mode_dir / "predictions.json", "w", encoding="utf-8") as f:
            json.dump(preds, f, indent=2)
        result["predictions"] = preds  # restore

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30} {'Accuracy':>15}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['experiment']:<30} {r['accuracy_ci']:>15}")

    # Save overall summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_samples, "seed": seed,
        "results": [{k: v for k, v in r.items() if k != "predictions"}
                     for r in all_results],
    }
    with open(out / "overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
