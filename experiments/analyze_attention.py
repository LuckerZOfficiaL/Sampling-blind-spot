#!/usr/bin/env python3
"""
Attention pattern analysis: baseline vs activation-grafted generation.

For each example, captures per-layer attention statistics for both a baseline run
and a grafted run, allowing direct comparison of how grafting affects attention.

What is captured
----------------
  prefill_attn_received[layer, token]:
      During the prompt forward pass (prefill), the mean attention weight received
      by each prompt token, averaged over all heads AND all query positions.
      Shape: [n_layers, prompt_len]

  gen_attn_on_prompt[step, layer, token]:
      During the generation of the (step+1)-th output token, the mean attention
      weight placed on each prompt token by that new query, averaged over heads.
      Shape: [n_gen_steps, n_layers, prompt_len]

  gen_mass_on_prompt[step, layer]:
      Total attention mass placed on ALL prompt tokens at each generation step.
      Complement (1 - this) is mass placed on already-generated tokens.
      Shape: [n_gen_steps, n_layers]

How grafting interacts with attention
--------------------------------------
  Grafting in SINGLE_SHOT mode fires a forward hook on the POST-MLP output of
  layer_b during the PREFILL pass only. This means:
    - Layers 0 .. layer_b-1 : identical attention to baseline (hook not yet fired)
    - Layer layer_b          : identical INTERNAL attention to baseline (hook fires
                               AFTER the attention op completes), but the output
                               hidden state passed to layer_b+1 is replaced
    - Layers layer_b+1 .. N  : different attention because their input is changed

  During generation, the hook does NOT fire again (SINGLE_SHOT). However, the KV
  cache was built during the grafted prefill, so the K/V tensors in layers
  layer_b+1..N are already modified — generation attention there will differ.

Requirements
------------
  The model MUST be loaded with attn_implementation="eager" (done automatically
  by this script).  Flash-attention-2 and SDPA do not materialise or return the
  attention weight matrix, so output_attentions=True would silently fall back or
  raise an error.

Usage
-----
    python experiments/analyze_attention.py \\
        --model-b $CKPTS/Qwen2.5-3B-Instruct \\
        --graft-vector zero \\
        --layer-b 26 \\
        --graft-position -1 \\
        --benchmark gsm8k \\
        --n-samples 5 \\
        --n-gen-steps 30 \\
        --output-dir attention_analysis/

    # Also capture a real model-A source:
    python experiments/analyze_attention.py \\
        --model-b $CKPTS/Qwen2.5-3B-Instruct \\
        --model-a $CKPTS/nomadicsynth3B \\
        --layer-b 26 --layer-a 26 \\
        --graft-vector model_a \\
        --output-dir attention_analysis/nomadic_vs_zero

Output
------
  {output_dir}/
      example_{i:03d}/
          baseline.npz   – prefill_attn_received, gen_attn_on_prompt,
                           gen_mass_on_prompt, token_ids
          grafted.npz    – same tensors for the grafted run
          meta.json      – prompt text, token strings, generated outputs
      summary.json       – per-example metadata + correctness
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.reasoning_tasks import (
    load_gsm8k_subset,
    load_medmcqa_subset,
    load_math_subset,
    load_mmlu_pro_subset,
)
from src.models.layer_utils import (
    get_transformer_layers,
    unpack_hook_output,
    repack_hook_output,
)


# =============================================================================
# Prompt formatting (inlined to keep script self-contained)
# =============================================================================

def _apply_chat(messages, tokenizer):
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return messages[-1]["content"]


def format_gsm8k(question: str, tokenizer) -> str:
    return _apply_chat(
        [{"role": "user", "content": (
            "Solve this math problem step by step. "
            "At the end, provide your final answer as a number after '#### '.\n\n"
            f"Problem: {question}"
        )}],
        tokenizer,
    )


def format_mcq(ex, tokenizer, choices_labels=None) -> str:
    if choices_labels is None:
        choices_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(
        f"{l}. {c}" for l, c in zip(choices_labels, ex.choices)
    )
    content = (
        "The following is a multiple choice question. "
        "Think step by step, then wrap your final answer as <answer>X</answer>.\n\n"
        f"Question: {ex.question}\n{choices_text}\n\nAnswer:"
    )
    return _apply_chat([{"role": "user", "content": content}], tokenizer)


def format_math(ex, tokenizer) -> str:
    return _apply_chat(
        [{"role": "user", "content": (
            "Solve the following competition math problem step by step. "
            r"Put your final answer inside \boxed{}." + f"\n\nProblem: {ex.question}"
        )}],
        tokenizer,
    )


BENCHMARK_FORMATTERS = {
    "gsm8k":    lambda ex, tok: format_gsm8k(ex.question, tok),
    "medmcqa":  lambda ex, tok: format_mcq(ex, tok),
    "math":     lambda ex, tok: format_math(ex, tok),
    "mmlu_pro": lambda ex, tok: format_mcq(ex, tok, choices_labels=list("ABCDEFGHIJ"[:len(ex.choices)])),
}

BENCHMARK_LOADERS = {
    "gsm8k":    load_gsm8k_subset,
    "medmcqa":  load_medmcqa_subset,
    "math":     load_math_subset,
    "mmlu_pro": load_mmlu_pro_subset,
}


# =============================================================================
# Graft vector construction
# =============================================================================

def _capture_b_hidden(model_b, inputs, layer_b: int, graft_pos: int) -> torch.Tensor:
    """Return the hidden state at (layer_b, graft_pos) of model_b for these inputs."""
    captured = {}

    def _hook(_m, _inp, out):
        h, _ = unpack_hook_output(out)
        seq_len = h.shape[1]
        pos = graft_pos if graft_pos >= 0 else seq_len + graft_pos
        captured["h"] = h[0, pos, :].detach().cpu().float()

    layers = get_transformer_layers(model_b)
    handle = layers[layer_b].register_forward_hook(_hook)
    with torch.no_grad():
        model_b(**inputs)
    handle.remove()
    return captured["h"]  # [d_model]


def build_graft_vector(
    graft_type: str,
    model_b,
    inputs_b,
    layer_b: int,
    graft_pos: int,
    model_a=None,
    layer_a: int = 26,
    prompt: str = None,
    tokenizer_a=None,
) -> torch.Tensor:
    """
    Return a [d_model] float32 CPU tensor to inject at (layer_b, graft_pos).

    graft_type:
        "zero"       – all-zeros vector
        "random"     – random unit vector scaled to match B's activation norm
        "model_a"    – POST-MLP hidden state of model_a at layer_a at graft_pos

    For "model_a", the prompt is re-tokenized with tokenizer_a (which may differ
    from tokenizer_b for cross-architecture pairs).
    """
    d_model = model_b.config.hidden_size

    if graft_type == "zero":
        return torch.zeros(d_model)

    if graft_type in ("random", "random_unit"):
        b_hidden = _capture_b_hidden(model_b, inputs_b, layer_b, graft_pos)
        b_norm = b_hidden.norm().item()
        v = torch.randn(d_model)
        v = v / v.norm() * b_norm
        return v

    if graft_type == "model_a":
        if model_a is None:
            raise ValueError("--model-a path required when --graft-vector model_a")
        if tokenizer_a is None or prompt is None:
            raise ValueError("tokenizer_a and prompt required for model_a graft vector")
        captured = {}

        def _hook_a(_m, _inp, out):
            h, _ = unpack_hook_output(out)
            seq_len = h.shape[1]
            pos = graft_pos if graft_pos >= 0 else seq_len + graft_pos
            captured["act"] = h[0, pos, :].detach().cpu().float()

        layers_a = get_transformer_layers(model_a)
        handle = layers_a[layer_a].register_forward_hook(_hook_a)
        # Tokenize with model_a's own tokenizer (may differ from model_b's)
        inputs_a = tokenizer_a(
            prompt, return_tensors="pt", truncation=True
        ).to(next(model_a.parameters()).device)
        with torch.no_grad():
            model_a(**inputs_a)
        handle.remove()
        return captured["act"]  # [d_model]

    raise ValueError(f"Unknown graft_type: {graft_type!r}. "
                     f"Choose from: zero, random, model_a")


# =============================================================================
# Grafting hook
# =============================================================================

def make_single_shot_hook(graft_vector: torch.Tensor, graft_pos: int):
    """
    Forward hook that replaces hidden[:, graft_pos, :] with graft_vector on the
    FIRST forward pass only (SINGLE_SHOT semantics).

    graft_vector: [d_model] CPU tensor (will be moved to the layer's device at call time)
    graft_pos:    token position, negative values count from end (-1 = last)
    """
    fired = [False]
    vec = graft_vector.clone()  # local copy

    def hook(_module, _input, output):
        if fired[0]:
            return output
        fired[0] = True

        hidden_states, rest = unpack_hook_output(output)
        seq_len = hidden_states.shape[1]
        pos = graft_pos if graft_pos >= 0 else seq_len + graft_pos

        if pos < 0 or pos >= seq_len:
            # Position out of range (e.g. graft_pos=-2 on a 1-token input during
            # decode steps — shouldn't happen in SINGLE_SHOT but guard anyway)
            return output

        hidden_states = hidden_states.clone()
        hidden_states[:, pos, :] = vec.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        return repack_hook_output(hidden_states, rest)

    return hook


# =============================================================================
# Attention statistics extraction
# =============================================================================

def _summarise_attentions(
    raw_attentions,        # tuple of tuples from model.generate()
    prompt_len: int,
    graft_pos: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Process the raw attention tensors returned by model.generate() with
    output_attentions=True.

    HuggingFace includes the PREFILL pass as the first element of the returned
    tuple (q_len == prompt_len), followed by one element per generated token
    (q_len == 1, kv_len grows by 1 each step).

    Returns
    -------
    prefill_attn_received : [n_layers, prompt_len]
        Mean attention weight RECEIVED by each prompt token during prefill,
        averaged over all heads and all query positions.

        Note — causal-mask bias: due to the causal mask, token 0 is attended to
        by all query positions while the last token is attended to only by itself.
        When averaging over query rows, earlier tokens accumulate more "received"
        attention. This bias is identical in baseline and grafted runs, so the
        DIFFERENCE between runs is unaffected, but absolute values should not be
        compared across token positions.

    gen_attn_on_prompt : [n_gen_steps, n_layers, prompt_len]
        Mean attention weight placed on each prompt token by each newly generated
        token's query, averaged over heads.

    gen_mass_on_prompt : [n_gen_steps, n_layers]
        Total attention mass placed on all prompt tokens at each generation step
        (sum of gen_attn_on_prompt[t, l, :] over the token dimension).

    prefill_attn_from_graft : [n_layers, prompt_len] or None
        ROW of the graft position in the prefill attention matrix (mean over heads).
        What the grafted token attends TO. Zero for layers <= layer_b because
        the hook fires after the attention op within that layer.

    prefill_attn_to_graft : [n_layers, prompt_len] or None
        COLUMN of the graft position (mean over heads).
        How much each token attends TO the grafted token.
        No dilution — directly shows the K-vector effect of grafting.
    """
    n_layers = len(raw_attentions[0])

    # ── Detect whether raw_attentions[0] is the prefill or first decode step ──
    # For prefill: q_len == kv_len == prompt_len  (full self-attention over the prompt)
    # For decode step 0: q_len == 1, kv_len == prompt_len + 1
    # Checking both dimensions avoids ambiguity when prompt_len == 1.
    step0_shape = raw_attentions[0][0].shape   # [batch, heads, q_len, kv_len]
    step0_q_len  = step0_shape[2]
    step0_kv_len = step0_shape[3]

    if step0_q_len == prompt_len and step0_kv_len == prompt_len:
        prefill_attns = raw_attentions[0]   # n_layers of [1, heads, PL, PL]
        decode_attns  = raw_attentions[1:]  # remaining steps
    else:
        # Prefill not included (shouldn't happen with standard generate but handle gracefully)
        prefill_attns = None
        decode_attns  = raw_attentions

    # ── Prefill summary ───────────────────────────────────────────────────────
    if prefill_attns is not None:
        prefill_arr = np.zeros((n_layers, prompt_len), dtype=np.float32)

        # Resolve graft position for row/column extraction
        if graft_pos is not None:
            pos = graft_pos if graft_pos >= 0 else prompt_len + graft_pos
            prefill_from = np.zeros((n_layers, prompt_len), dtype=np.float32)
            prefill_to   = np.zeros((n_layers, prompt_len), dtype=np.float32)
        else:
            pos = None
            prefill_from = prefill_to = None

        for l, layer_attn in enumerate(prefill_attns):
            # [1, heads, prompt_len, prompt_len] → [heads, PL, PL]
            attn = layer_attn[0].float()          # [heads, PL, PL]
            attn_mean = attn.mean(dim=0)          # [PL, PL]  (mean over heads)
            # Mean over query rows → [PL]  (attention received per token)
            prefill_arr[l] = attn_mean.mean(dim=0).cpu().numpy()
            if pos is not None:
                # Row: what the grafted token attends to  [PL]
                prefill_from[l] = attn_mean[pos, :].cpu().numpy()
                # Column: how much each token attends to the grafted token  [PL]
                prefill_to[l]   = attn_mean[:, pos].cpu().numpy()
    else:
        # Fall back: fill with NaN so callers know it wasn't captured
        prefill_arr  = np.full((n_layers, prompt_len), np.nan, dtype=np.float32)
        prefill_from = prefill_to = None

    # ── Generation step summary ───────────────────────────────────────────────
    n_gen = len(decode_attns)
    gen_attn  = np.zeros((n_gen, n_layers, prompt_len), dtype=np.float32)
    gen_mass  = np.zeros((n_gen, n_layers), dtype=np.float32)

    for t, step_attns in enumerate(decode_attns):
        for l, layer_attn in enumerate(step_attns):
            # [1, heads, 1, kv_len] where kv_len = prompt_len + t + 1 (or +t if
            # the prefill was NOT in attentions[0])
            attn = layer_attn[0, :, 0, :].float()  # [heads, kv_len]
            attn_mean = attn.mean(dim=0).cpu().numpy()  # [kv_len]
            attn_on_prompt = attn_mean[:prompt_len]
            gen_attn[t, l] = attn_on_prompt
            gen_mass[t, l] = attn_on_prompt.sum()

    return prefill_arr, gen_attn, gen_mass, prefill_from, prefill_to


def run_single(
    model,
    tokenizer,
    inputs: dict,
    n_gen_steps: int,
    graft_vector: Optional[torch.Tensor] = None,
    layer_b: int = 26,
    graft_pos: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], torch.Tensor]:
    """
    Run model.generate() with attention logging.

    If graft_vector is not None, a SINGLE_SHOT grafting hook is registered on
    layer_b before calling generate(). The hook fires only on the prefill pass.

    Returns
    -------
    prefill_attn_received   : [n_layers, prompt_len]
    gen_attn_on_prompt      : [n_gen_steps, n_layers, prompt_len]
    gen_mass_on_prompt      : [n_gen_steps, n_layers]
    prefill_attn_from_graft : [n_layers, prompt_len]  (row at graft_pos, mean over heads)
    prefill_attn_to_graft   : [n_layers, prompt_len]  (column at graft_pos, mean over heads)
    generated_ids           : 1-D token ID tensor (new tokens only)
    """
    prompt_len = inputs["input_ids"].shape[1]

    handle = None
    if graft_vector is not None:
        hook = make_single_shot_hook(graft_vector, graft_pos)
        layers = get_transformer_layers(model)
        handle = layers[layer_b].register_forward_hook(hook)

    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=n_gen_steps,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if handle is not None:
            handle.remove()

    generated_ids = out.sequences[0][prompt_len:]

    prefill_arr, gen_attn, gen_mass, prefill_from, prefill_to = _summarise_attentions(
        out.attentions, prompt_len, graft_pos=graft_pos,
    )

    return prefill_arr, gen_attn, gen_mass, prefill_from, prefill_to, generated_ids


# =============================================================================
# Option 2 — Teacher-forced forward pass
# =============================================================================

def run_teacher_forced(
    model,
    prompt_inputs: dict,
    gen_ids_base: torch.Tensor,
    graft_vector: torch.Tensor,
    layer_b: int,
    graft_pos: int,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Run a full forward pass on (prompt + baseline_output) for BOTH baseline and
    grafted models.  Because both passes process the IDENTICAL token sequence,
    any difference in the attention matrices is 100% attributable to grafting.

    The grafting hook fires once during the grafted forward pass at
    (layer_b, graft_pos), modifying hidden states for all subsequent layers
    AND all subsequent token positions (including the generated tokens).

    Parameters
    ----------
    prompt_inputs  : tokenized prompt (already on model.device)
    gen_ids_base   : 1-D tensor of token IDs generated by the baseline run
                     (used to build the full sequence; no re-tokenization needed)
    graft_vector   : [d_model] CPU tensor to inject
    layer_b, graft_pos : grafting coordinates

    Returns
    -------
    attn_base      : [n_layers, full_seq_len, full_seq_len]  mean-over-heads
    attn_diff      : [n_layers, full_seq_len, full_seq_len]  grafted − baseline
    prompt_len     : int  (boundary between prompt and generated tokens)
    full_token_ids : [full_seq_len] numpy array
    """
    # Build full token sequence: prompt IDs + baseline generated IDs
    prompt_ids = prompt_inputs["input_ids"]                          # [1, prompt_len]
    gen_ids    = gen_ids_base.unsqueeze(0).to(prompt_ids.device)     # [1, gen_len]
    full_ids   = torch.cat([prompt_ids, gen_ids], dim=1)             # [1, full_seq_len]
    full_inputs = {
        "input_ids":      full_ids,
        "attention_mask": torch.ones_like(full_ids),
    }

    prompt_len   = prompt_ids.shape[1]
    full_seq_len = full_ids.shape[1]
    n_layers     = model.config.num_hidden_layers

    def _extract(out) -> np.ndarray:
        """Mean-over-heads attention for each layer → [n_layers, S, S]."""
        result = np.zeros((n_layers, full_seq_len, full_seq_len), dtype=np.float32)
        for l, la in enumerate(out.attentions):
            result[l] = la[0].float().mean(dim=0).cpu().numpy()
        return result

    # ── Baseline forward pass ─────────────────────────────────────────────────
    with torch.no_grad():
        out_base = model(**full_inputs, output_attentions=True)
    attn_base = _extract(out_base)
    del out_base

    # ── Grafted forward pass ──────────────────────────────────────────────────
    # Resolve graft_pos relative to PROMPT length, not full sequence length.
    # With graft_pos=-1 and full_seq_len=prompt+gen, make_single_shot_hook would
    # otherwise resolve -1 to the last generated token instead of the last prompt
    # token. Convert to absolute index here so the hook grafts at the right place.
    resolved_pos = graft_pos if graft_pos >= 0 else prompt_len + graft_pos
    hook   = make_single_shot_hook(graft_vector, resolved_pos)
    layers = get_transformer_layers(model)
    handle = layers[layer_b].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out_graft = model(**full_inputs, output_attentions=True)
    finally:
        handle.remove()
    attn_graft = _extract(out_graft)
    del out_graft

    attn_diff = (attn_graft - attn_base).astype(np.float32)

    return attn_base, attn_diff, prompt_len, full_ids[0].cpu().numpy()


# =============================================================================
# Option 3 — Hidden state divergence across layers
# =============================================================================

def collect_hidden_trajectories(
    model,
    inputs: dict,
    graft_vector: torch.Tensor,
    layer_b: int,
    graft_pos: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Capture the hidden state at graft_pos after every transformer layer for
    both a baseline and a grafted forward pass.

    The grafting hook is registered on layer_b BEFORE the per-layer capture
    hooks, so at layer_b the capture records the POST-GRAFT hidden state (the
    injected vector itself).  For layers > layer_b, the captured state is the
    network's response to the injection.

    Parameters
    ----------
    inputs       : tokenized prompt (already on model.device)
    graft_vector : [d_model] CPU tensor
    layer_b, graft_pos : grafting coordinates

    Returns
    -------
    hs_base  : [n_layers, d_model] float32
    hs_graft : [n_layers, d_model] float32

    Derived quantities (compute in analysis):
        cos_sim = (hs_base * hs_graft).sum(-1) / (
            np.linalg.norm(hs_base, axis=-1) * np.linalg.norm(hs_graft, axis=-1)
        )
        l2_dist = np.linalg.norm(hs_graft - hs_base, axis=-1)
    """
    n_layers = model.config.num_hidden_layers
    d_model  = model.config.hidden_size
    layers   = get_transformer_layers(model)

    # Resolve position once (sequence length is the same for both passes)
    seq_len = inputs["input_ids"].shape[1]
    pos = graft_pos if graft_pos >= 0 else seq_len + graft_pos

    def make_capture(storage, idx):
        def hook(_m, _inp, out):
            h, _ = unpack_hook_output(out)
            storage[idx] = h[0, pos, :].detach().cpu().float().numpy()
        return hook

    # ── Baseline ──────────────────────────────────────────────────────────────
    hs_base = np.zeros((n_layers, d_model), dtype=np.float32)
    handles = [layers[l].register_forward_hook(make_capture(hs_base, l))
               for l in range(n_layers)]
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()

    # ── Grafted ───────────────────────────────────────────────────────────────
    # Register graft hook FIRST so it fires before the capture hook at layer_b.
    # Layer_b capture then records the injected vector, not the original output.
    hs_graft    = np.zeros((n_layers, d_model), dtype=np.float32)
    graft_hook  = make_single_shot_hook(graft_vector, graft_pos)
    graft_hdl   = layers[layer_b].register_forward_hook(graft_hook)
    handles     = [layers[l].register_forward_hook(make_capture(hs_graft, l))
                   for l in range(n_layers)]
    with torch.no_grad():
        model(**inputs)
    graft_hdl.remove()
    for h in handles:
        h.remove()

    return hs_base, hs_graft


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare attention patterns baseline vs grafted generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Zero-vector grafting on GSM8K, last-token position, 5 examples, 30 gen steps
  python experiments/analyze_attention.py \\
      --model-b $CKPTS/Qwen2.5-3B-Instruct \\
      --graft-vector zero \\
      --layer-b 26 --graft-position -1 \\
      --benchmark gsm8k --n-samples 5 \\
      --n-gen-steps 30 \\
      --output-dir attention_analysis/zero_last

  # Random-vector grafting on first-token position
  python experiments/analyze_attention.py \\
      --model-b $CKPTS/Qwen2.5-3B-Instruct \\
      --graft-vector random --graft-position 0 \\
      --layer-b 26 --n-gen-steps 30 \\
      --output-dir attention_analysis/random_first

  # Real model-A source (nomadicsynth)
  python experiments/analyze_attention.py \\
      --model-b $CKPTS/Qwen2.5-3B-Instruct \\
      --model-a  $CKPTS/nomadicsynth3B \\
      --layer-b 26 --layer-a 26 \\
      --graft-vector model_a \\
      --output-dir attention_analysis/nomadic

Notes
-----
  - output_attentions=True requires attn_implementation="eager" (no flash-attn).
    This script loads the model with eager attention automatically.
  - Prefer small --n-gen-steps (20–50): each step stores n_layers attention
    tensors in memory before summarisation.
  - Use --n-samples 5–20 for a quick analysis run.
""",
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    p.add_argument("--model-b", required=True,
                   help="Path or HuggingFace name of the target/receiver model.")
    p.add_argument("--model-a", default=None,
                   help="Path or name of the source model (required only when "
                        "--graft-vector model_a).")
    p.add_argument("--layer-b", type=int, default=26,
                   help="Layer index in model-b where grafting is applied (default: 26).")
    p.add_argument("--layer-a", type=int, default=26,
                   help="Layer index in model-a to extract from (default: 26).")

    # ── Grafting ───────────────────────────────────────────────────────────────
    p.add_argument("--graft-vector", default="zero",
                   choices=["zero", "random", "model_a"],
                   help="Type of graft vector to use (default: zero).")
    p.add_argument("--graft-position", type=int, default=-1,
                   help="Token position to graft at. -1 = last token (default), "
                        "0 = first token, -2 = penultimate.")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    p.add_argument("--benchmark", default="gsm8k",
                   choices=list(BENCHMARK_LOADERS.keys()),
                   help="Benchmark to sample from (default: gsm8k).")
    p.add_argument("--n-samples", type=int, default=10,
                   help="Number of examples to analyse (default: 10).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for example sampling (default: 42).")

    # ── Generation ────────────────────────────────────────────────────────────
    p.add_argument("--n-gen-steps", type=int, default=30,
                   help="Number of generation steps for which to capture attention "
                        "(default: 30). Keep small (≤50) to limit memory usage.")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output-dir", required=True,
                   help="Directory to write results to.")

    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Load model-b with EAGER attention ─────────────────────────────────────
    print(f"Loading model-b (eager attn): {args.model_b}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_b, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_b = AutoModelForCausalLM.from_pretrained(
        args.model_b,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model_b.eval()
    print(f"  Layers: {model_b.config.num_hidden_layers}, "
          f"d_model: {model_b.config.hidden_size}, "
          f"heads: {model_b.config.num_attention_heads}")

    # ── Load model-a if needed ────────────────────────────────────────────────
    model_a = None
    tokenizer_a = None
    if args.graft_vector == "model_a":
        if args.model_a is None:
            raise ValueError("--model-a is required when --graft-vector model_a")
        print(f"Loading model-a (eager attn): {args.model_a}")
        tokenizer_a = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
        if tokenizer_a.pad_token is None:
            tokenizer_a.pad_token = tokenizer_a.eos_token
            tokenizer_a.pad_token_id = tokenizer_a.eos_token_id
        model_a = AutoModelForCausalLM.from_pretrained(
            args.model_a,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model_a.eval()

    # ── Set global seed for reproducibility (random graft vectors, etc.) ─────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load examples ─────────────────────────────────────────────────────────
    print(f"Loading {args.n_samples} examples from {args.benchmark}…")
    loader = BENCHMARK_LOADERS[args.benchmark]
    examples = loader(n_samples=args.n_samples, seed=args.seed)
    formatter = BENCHMARK_FORMATTERS[args.benchmark]
    print(f"  Loaded {len(examples)} examples")

    # ── Save config snapshot ───────────────────────────────────────────────────
    config_snapshot = vars(args)
    (out_root / "config.json").write_text(
        json.dumps(config_snapshot, indent=2), encoding="utf-8"
    )

    summary = []

    for i, ex in enumerate(tqdm(examples, desc="Analysing")):
        ex_dir = out_root / f"example_{i:03d}"
        ex_dir.mkdir(exist_ok=True)

        # Format and tokenize
        prompt = formatter(ex, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(
            model_b.device
        )
        prompt_len = inputs["input_ids"].shape[1]
        token_strings = [
            tokenizer.decode([tid]) for tid in inputs["input_ids"][0].tolist()
        ]

        # ── Baseline run ──────────────────────────────────────────────────────
        pre_base, gen_base, mass_base, pre_from_base, pre_to_base, gen_ids_base = run_single(
            model_b, tokenizer, inputs,
            n_gen_steps=args.n_gen_steps,
            graft_pos=args.graft_position,
        )
        baseline_output = tokenizer.decode(gen_ids_base, skip_special_tokens=True).strip()

        np.savez_compressed(
            ex_dir / "baseline.npz",
            prefill_attn_received=pre_base,        # [n_layers, prompt_len]
            prefill_attn_from_graft=pre_from_base, # [n_layers, prompt_len]  row at graft_pos
            prefill_attn_to_graft=pre_to_base,     # [n_layers, prompt_len]  col at graft_pos
            gen_attn_on_prompt=gen_base,            # [n_gen_steps, n_layers, prompt_len]
            gen_mass_on_prompt=mass_base,           # [n_gen_steps, n_layers]
            token_ids=inputs["input_ids"][0].cpu().numpy(),
        )

        # ── Build graft vector ────────────────────────────────────────────────
        graft_vec = build_graft_vector(
            graft_type=args.graft_vector,
            model_b=model_b,
            inputs_b=inputs,
            layer_b=args.layer_b,
            graft_pos=args.graft_position,
            model_a=model_a,
            layer_a=args.layer_a,
            prompt=prompt,
            tokenizer_a=tokenizer_a,
        )

        # ── Grafted run ───────────────────────────────────────────────────────
        pre_graft, gen_graft, mass_graft, pre_from_graft, pre_to_graft, gen_ids_graft = run_single(
            model_b, tokenizer, inputs,
            n_gen_steps=args.n_gen_steps,
            graft_vector=graft_vec,
            layer_b=args.layer_b,
            graft_pos=args.graft_position,
        )
        grafted_output = tokenizer.decode(gen_ids_graft, skip_special_tokens=True).strip()

        np.savez_compressed(
            ex_dir / "grafted.npz",
            prefill_attn_received=pre_graft,
            prefill_attn_from_graft=pre_from_graft,
            prefill_attn_to_graft=pre_to_graft,
            gen_attn_on_prompt=gen_graft,
            gen_mass_on_prompt=mass_graft,
            token_ids=inputs["input_ids"][0].cpu().numpy(),
        )

        # ── Teacher-forced forward pass (Option 2) ───────────────────────────
        # Both runs process the EXACT same token sequence (prompt + baseline output).
        # Any attention difference is 100% due to grafting, with no token confound.
        attn_base_tf, attn_diff_tf, tf_prompt_len, tf_token_ids = run_teacher_forced(
            model_b, inputs, gen_ids_base,
            graft_vector=graft_vec,
            layer_b=args.layer_b,
            graft_pos=args.graft_position,
        )
        np.savez_compressed(
            ex_dir / "teacher_forced.npz",
            attn_base=attn_base_tf,      # [n_layers, full_seq_len, full_seq_len]
            attn_diff=attn_diff_tf,      # [n_layers, full_seq_len, full_seq_len] grafted−baseline
            prompt_len=np.array(tf_prompt_len),
            full_token_ids=tf_token_ids, # [full_seq_len]
        )

        # ── Hidden state trajectory (Option 3) ───────────────────────────────
        # Capture hidden state at graft_pos after every layer.
        # Shows how the injected perturbation propagates or decays layer by layer.
        hs_base, hs_graft = collect_hidden_trajectories(
            model_b, inputs,
            graft_vector=graft_vec,
            layer_b=args.layer_b,
            graft_pos=args.graft_position,
        )
        # Precompute cosine similarity and L2 distance
        norms_base  = np.linalg.norm(hs_base,  axis=-1)
        norms_graft = np.linalg.norm(hs_graft, axis=-1)
        dot         = (hs_base * hs_graft).sum(axis=-1)
        cos_sim     = dot / (norms_base * norms_graft + 1e-8)
        l2_dist     = np.linalg.norm(hs_graft - hs_base, axis=-1)
        np.savez_compressed(
            ex_dir / "hidden_trajectory.npz",
            hs_base=hs_base,       # [n_layers, d_model]
            hs_graft=hs_graft,     # [n_layers, d_model]
            cos_sim=cos_sim,       # [n_layers]  1.0 = identical, <1.0 = diverged
            l2_dist=l2_dist,       # [n_layers]  0.0 = identical, larger = more diverged
        )

        # ── Per-example metadata ──────────────────────────────────────────────
        meta = {
            "example_index": i,
            "id": str(ex.id),
            "prompt_len": prompt_len,
            "token_strings": token_strings,
            "baseline_output": baseline_output,
            "grafted_output":  grafted_output,
            "n_gen_steps_actual_baseline": int(gen_base.shape[0]),
            "n_gen_steps_actual_grafted":  int(gen_graft.shape[0]),
            "graft_vector_type": args.graft_vector,
            "layer_b": args.layer_b,
            "graft_position": args.graft_position,
        }
        (ex_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # ── Quick console diff ────────────────────────────────────────────────
        # Compute mean absolute difference in prefill attention by layer region.
        # Guard against empty slices when layer_b is 0 or the last layer.
        before_slice = pre_graft[:args.layer_b]  - pre_base[:args.layer_b]
        after_slice  = pre_graft[args.layer_b+1:] - pre_base[args.layer_b+1:]
        diff_before = float(np.abs(before_slice).mean()) if before_slice.size > 0 else 0.0
        diff_at     = float(np.abs(pre_graft[args.layer_b] - pre_base[args.layer_b]).mean())
        diff_after  = float(np.abs(after_slice).mean())  if after_slice.size  > 0 else 0.0
        tqdm.write(
            f"  [{i:3d}] prompt_len={prompt_len:4d} | "
            f"prefill Δattn: before_L{args.layer_b}={diff_before:.4f}, "
            f"at_L{args.layer_b}={diff_at:.4f}, "
            f"after_L{args.layer_b}={diff_after:.4f}"
        )

        summary.append({
            "example_index": i,
            "id": str(ex.id),
            "prompt_len": prompt_len,
            "n_gen_steps_actual_baseline": int(gen_base.shape[0]),
            "n_gen_steps_actual_grafted":  int(gen_graft.shape[0]),
            "prefill_mean_abs_diff_before_graft_layer": diff_before,
            "prefill_mean_abs_diff_at_graft_layer":     diff_at,
            "prefill_mean_abs_diff_after_graft_layer":  diff_after,
            "gen_mean_mass_on_prompt_baseline": float(mass_base.mean()),
            "gen_mean_mass_on_prompt_grafted":  float(mass_graft.mean()),
        })

    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to {out_root}/")
    print("  Per example:")
    print("    baseline.npz          — prefill_attn_received, prefill_attn_from_graft,")
    print("                            prefill_attn_to_graft, gen_attn_on_prompt,")
    print("                            gen_mass_on_prompt, token_ids")
    print("    grafted.npz           — same arrays for the grafted run")
    print("    teacher_forced.npz    — attn_base, attn_diff [n_layers, S, S],")
    print("                            prompt_len, full_token_ids")
    print("                            (both runs on IDENTICAL token sequence)")
    print("    hidden_trajectory.npz — hs_base, hs_graft [n_layers, d_model],")
    print("                            cos_sim, l2_dist [n_layers]")
    print("    meta.json")
    print("  Overall: summary.json, config.json")
    print("\nKey arrays:")
    print("  prefill_attn_from_graft[l, k] = what grafted token attends to at layer l (row)")
    print("  prefill_attn_to_graft[l, k]   = how much token k attends to graft pos (col)")
    print("  attn_diff[l, q, k]            = grafted − baseline, same-token comparison")
    print("  cos_sim[l]                    = cosine similarity at graft_pos, 1.0=identical")


if __name__ == "__main__":
    main()
