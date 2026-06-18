"""Shared inference utilities for chat prompt formatting and text generation."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer


def build_chat_prompt(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build a chat prompt string from messages using the tokenizer's chat template.

    Args:
        tokenizer: HuggingFace tokenizer with chat template support
        messages: List of message dicts with 'role' and 'content' keys.
            May also contain 'tool_calls' for assistant messages (native tool calling).
        tools: Optional list of tool schemas for native tool calling.
            Each dict should follow the OpenAI-style format:
            {'type': 'function', 'function': {'name': ..., 'description': ..., 'parameters': ...}}

    Returns:
        Formatted prompt string
    """
    try:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:  # pylint: disable=broad-exception-caught
        # Fallback: manual Llama-style formatting
        prompt = "<|begin_of_text|>"
        for msg in messages:
            role = msg["role"]
            # Map tool/ipython roles to ipython header for Llama format
            if role in ("tool", "ipython"):
                role = "ipython"
            # Handle assistant messages with tool_calls
            if msg.get("tool_calls"):
                prompt += (
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                    f"<|python_tag|>"
                )
                import json as _json  # pylint: disable=import-outside-toplevel
                for tc in msg["tool_calls"]:
                    func = tc.get("function", tc)
                    call_obj = {"name": func["name"], "parameters": func.get("arguments", func.get("parameters", {}))}
                    prompt += _json.dumps(call_obj)
                prompt += "<|eot_id|>"
            else:
                content = msg.get("content", "")
                prompt += (
                    f"<|start_header_id|>{role}<|end_header_id|>"
                    f"\n\n{content.strip()}<|eot_id|>"
                )
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    **generation_kwargs,
) -> str:
    """Generate a single response from a model given a prompt string.

    Args:
        model: HuggingFace causal language model
        tokenizer: Corresponding tokenizer
        prompt: Formatted prompt string
        max_new_tokens: Maximum number of tokens to generate
        device: Device to place inputs on
        **generation_kwargs: Extra kwargs forwarded to model.generate

    Returns:
        Generated text (decoded, without the prompt)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "do_sample": False,
    }
    gen_kwargs.update(generation_kwargs)
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_response_with_tool_detection(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    **generation_kwargs,
) -> Tuple[str, bool]:
    """Generate a response and detect if it is a native Llama tool call.

    Checks the first generated token against the ``<|python_tag|>`` token ID
    to determine whether the model produced a native tool call.

    Args:
        model: HuggingFace causal language model
        tokenizer: Corresponding tokenizer
        prompt: Formatted prompt string
        max_new_tokens: Maximum number of tokens to generate
        device: Device to place inputs on
        **generation_kwargs: Extra kwargs forwarded to model.generate

    Returns:
        Tuple of (decoded_text, is_tool_call).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "do_sample": False,
    }
    gen_kwargs.update(generation_kwargs)
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

    # Check for <|python_tag|> as first token
    python_tag_id = tokenizer.convert_tokens_to_ids("<|python_tag|>")
    is_tool_call = (
        len(generated_ids) > 0
        and python_tag_id is not None
        and generated_ids[0].item() == python_tag_id
    )

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, is_tool_call


def tokenize_and_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
    **generation_kwargs,
) -> Tuple[str, Tensor, Tensor]:
    """Tokenize a prompt, generate, and return text plus raw tensors.

    This is a lower-level helper used by grafting engines that need
    access to the raw input IDs and generated IDs (e.g. for measuring
    output length).

    Args:
        model: HuggingFace causal language model.
        tokenizer: Corresponding tokenizer.
        prompt: Formatted prompt string.
        device: Device to place inputs on.
        max_new_tokens: Maximum number of tokens to generate.
        **generation_kwargs: Extra kwargs forwarded to model.generate.

    Returns:
        Tuple of (decoded_text, input_ids, generated_ids).
    """
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
    ).to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": (
            tokenizer.pad_token_id or tokenizer.eos_token_id
        ),
        "do_sample": False,
    }
    gen_kwargs.update(generation_kwargs)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_ids = inputs["input_ids"]
    generated_ids = outputs[0][input_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, input_ids, generated_ids


def extract_activation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    layer: int,
    extraction_point: "ExtractionPoint",
    device: torch.device,
    max_length: Optional[int] = None,
) -> Tuple[Tensor, int]:
    """Extract activation from a model at a specific layer.

    Tokenizes *text*, runs a forward pass through *model*, and returns
    the hidden-state vector at the last token position of the given layer.

    Args:
        model: HuggingFace causal language model.
        tokenizer: Corresponding tokenizer.
        text: Input text to extract from.
        layer: Layer index to extract.
        extraction_point: Where in the layer to extract (e.g. POST_MLP).
        device: Device to place inputs on.
        max_length: Optional max sequence length for tokenization.

    Returns:
        Tuple of (activation tensor ``[1, d_model]``, sequence_length).
    """
    from src.models.activation_extractor import ActivationExtractor  # pylint: disable=import-outside-toplevel

    tok_kwargs: Dict[str, Any] = {"return_tensors": "pt", "truncation": True}
    if max_length is not None:
        tok_kwargs["max_length"] = max_length

    inputs = tokenizer(text, **tok_kwargs).to(device)

    extractor = ActivationExtractor(
        model, [layer], extraction_point, token_position=-1,
    )

    with torch.no_grad():
        activations = extractor.extract(
            inputs["input_ids"], inputs.get("attention_mask"),
        )

    result = activations[layer]
    return result.tensor, result.sequence_length
