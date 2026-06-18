"""Shared utilities for transformer layers and hook output handling."""

from typing import Optional, Tuple

from torch import nn
from torch import Tensor
from transformers import PreTrainedModel


def get_transformer_layers(model: PreTrainedModel) -> nn.ModuleList:
    """
    Get the transformer layer modules from a HuggingFace model.

    Supports Llama, Qwen, Mistral, OLMo, GPT-2, GPT-NeoX, and encoder-decoder
    architectures.

    Args:
        model: A HuggingFace PreTrainedModel

    Returns:
        The nn.ModuleList of transformer layers

    Raises:
        ValueError: If the model architecture is not recognized
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Llama, Qwen, Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers  # GPT-NeoX
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        return model.model.decoder.layers  # Encoder-decoder models
    raise ValueError(
        f"Cannot find layers in model architecture: {type(model)}. "
        f"Supported: Llama, Qwen, Mistral, GPT-2, GPT-NeoX"
    )


def unpack_hook_output(output) -> Tuple[Tensor, Optional[tuple]]:
    """Unpack a forward hook's output into hidden_states and rest.

    Transformer layer forward hooks return either a single tensor
    or a tuple of (hidden_states, ...). This helper normalizes both
    cases.

    Args:
        output: The raw output from the forward hook.

    Returns:
        Tuple of (hidden_states, rest) where rest is None if output
        was a plain tensor.
    """
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def repack_hook_output(
    hidden_states: Tensor, rest: Optional[tuple]
) -> Tensor:
    """Re-pack hidden_states and rest into the original hook output format.

    Args:
        hidden_states: The (possibly modified) hidden states tensor.
        rest: The remaining tuple elements, or None.

    Returns:
        Repacked output matching the original format.
    """
    if rest is not None:
        return (hidden_states,) + rest
    return hidden_states
