"""Model loading utilities."""

from typing import Tuple, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.models.model_registry import get_model_spec, MODEL_REGISTRY
from src.utils.config import ModelConfig


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    return dtype_map.get(dtype_str, torch.float16)


def load_model_and_tokenizer(
    config: ModelConfig,
    device: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer from HuggingFace.

    Args:
        config: Model configuration
        device: Optional device override

    Returns:
        Tuple of (model, tokenizer)
    """
    # Get dtype
    torch_dtype = get_torch_dtype(config.torch_dtype)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.name,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model_kwargs = {
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": torch_dtype,
    }

    # Handle device mapping
    # Note: device_map="auto" doesn't work well with MPS, use explicit device instead
    if device is not None:
        model_kwargs["device_map"] = None
    elif device is None and torch.backends.mps.is_available():
        print("mps detected")
        # Auto-detect MPS on Apple Silicon
        device = "mps"
        model_kwargs["device_map"] = None
    elif config.device_map == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = config.device_map

    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        **model_kwargs,
    )

    # Move to device if specified
    if device is not None:
        model = model.to(device)

    # Set to eval mode
    model.eval()

    return model, tokenizer


def load_model_pair(
    config_a: ModelConfig,
    config_b: ModelConfig,
    device: Optional[str] = None,
) -> Tuple[
    PreTrainedModel, PreTrainedTokenizer,
    PreTrainedModel, PreTrainedTokenizer,
]:
    """
    Load a pair of models for grafting experiments.

    Args:
        config_a: Configuration for model A (sender)
        config_b: Configuration for model B (receiver)
        device: Optional device override

    Returns:
        Tuple of (model_a, tokenizer_a, model_b, tokenizer_b)
    """
    model_a, tokenizer_a = load_model_and_tokenizer(config_a, device)

    # If same model, reuse to save memory
    if config_a.name == config_b.name and config_a.revision == config_b.revision:
        model_b, tokenizer_b = model_a, tokenizer_a
    else:
        model_b, tokenizer_b = load_model_and_tokenizer(config_b, device)

    return model_a, tokenizer_a, model_b, tokenizer_b


def get_model_info(model: PreTrainedModel) -> dict:
    """
    Extract model information from a loaded model.

    Args:
        model: Loaded HuggingFace model

    Returns:
        Dictionary with model info
    """
    config = model.config

    return {
        "name": getattr(config, "_name_or_path", "unknown"),
        "d_model": getattr(config, "hidden_size", None),
        "n_layers": getattr(config, "num_hidden_layers", None),
        "n_heads": getattr(config, "num_attention_heads", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }


def validate_model_for_grafting(
    model: PreTrainedModel,
    model_name: str,
) -> bool:
    """
    Validate that a model can be used for activation grafting.

    Args:
        model: Loaded model
        model_name: Model name for registry lookup

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    # Check if in registry
    if model_name in MODEL_REGISTRY:
        spec = get_model_spec(model_name)
        info = get_model_info(model)

        # Validate dimensions match
        if info["d_model"] != spec.d_model:
            raise ValueError(
                f"Model d_model mismatch: expected {spec.d_model}, got {info['d_model']}"
            )
        if info["n_layers"] != spec.n_layers:
            raise ValueError(
                f"Model n_layers mismatch: expected {spec.n_layers}, got {info['n_layers']}"
            )

    # Check that we can find layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return True
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return True
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return True
    raise ValueError(
        f"Cannot find layer accessor for model architecture: {type(model)}"
    )

