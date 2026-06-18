"""Model registry with metadata for supported models."""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ModelSpec:
    """Specification for a supported model."""

    name: str
    d_model: int
    n_layers: int
    n_heads: int
    vocab_size: int
    tokenizer_name: str
    architecture: str  # "qwen2", "llama", "phi", "mistral", etc.
    rope_theta: float = 10000.0
    intermediate_size: Optional[int] = None

    @property
    def layer_accessor(self) -> str:
        """Path to access layers in the model."""
        if self.architecture in ["qwen2", "qwen3", "qwen3_5", "llama", "mistral", "olmo"]:
            return "model.layers"
        if self.architecture == "phi":
            return "model.layers"
        if self.architecture == "gpt2":
            return "transformer.h"
        if self.architecture == "gpt_neox":
            return "gpt_neox.layers"
        return "model.layers"  # Default


# Registry of supported models with their specifications
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # Qwen 2.5 family
    "Qwen/Qwen2.5-0.5B": ModelSpec(
        name="Qwen/Qwen2.5-0.5B",
        d_model=896,
        n_layers=24,
        n_heads=14,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-0.5B",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=4864,
    ),
    "Qwen/Qwen2.5-0.5B-Instruct": ModelSpec(
        name="Qwen/Qwen2.5-0.5B-Instruct",
        d_model=896,
        n_layers=24,
        n_heads=14,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=4864,
    ),
    "Qwen/Qwen2.5-1.5B": ModelSpec(
        name="Qwen/Qwen2.5-1.5B",
        d_model=1536,
        n_layers=28,
        n_heads=12,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-1.5B",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=8960,
    ),
    "Qwen/Qwen2.5-1.5B-Instruct": ModelSpec(
        name="Qwen/Qwen2.5-1.5B-Instruct",
        d_model=1536,
        n_layers=28,
        n_heads=12,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=8960,
    ),
    "Qwen/Qwen2.5-3B": ModelSpec(
        name="Qwen/Qwen2.5-3B",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "Qwen/Qwen2.5-3B-Instruct": ModelSpec(
        name="Qwen/Qwen2.5-3B-Instruct",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),

    # Qwen 2.5 3B fine-tunes
    "Qwen-2.5-3B-GSM325": ModelSpec(
        name="Qwen-2.5-3B-GSM325",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "Qwen-2.5-3B-Reasoning-yakuraku": ModelSpec(
        name="Qwen-2.5-3B-Reasoning-yakuraku",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "nomadicsynth3B": ModelSpec(
        name="nomadicsynth3B",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "bounhar3B": ModelSpec(
        name="bounhar3B",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "adamlucek3B": ModelSpec(
        name="adamlucek3B",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "coder3B": ModelSpec(
        name="coder3B",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "medmcqa3B": ModelSpec(
        name="medmcqa3B",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "qwen-3b-gsm8k": ModelSpec(
        name="qwen-3b-gsm8k",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "Qwen2.5-3B-GSM8k": ModelSpec(
        name="Qwen2.5-3B-GSM8k",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "Qwen2.5-3B-GSM8K-ft": ModelSpec(
        name="Qwen2.5-3B-GSM8K-ft",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "Qwen2.5-3B-GSM8K-ft-3eps": ModelSpec(
        name="Qwen2.5-3B-GSM8K-ft-3eps",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "banking3B": ModelSpec(
        name="banking3B",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "Qwen2.5-3B-Instruct-medmcqa": ModelSpec(
        name="Qwen2.5-3B-Instruct-medmcqa",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "Qwen2.5-3B-Instruct-mmlu": ModelSpec(
        name="Qwen2.5-3B-Instruct-mmlu",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "Qwen2.5-3B-Instruct_GSM8K-GRPO_16bit": ModelSpec(
        name="Qwen2.5-3B-Instruct_GSM8K-GRPO_16bit",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),
    "Qwen2.5-3B-NuminaMath-TIR": ModelSpec(
        name="Qwen2.5-3B-NuminaMath-TIR",
        d_model=2048,
        n_layers=36,
        n_heads=16,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        rope_theta=1000000.0,
        intermediate_size=11008,
    ),

    # ==========================================================================
    # EleutherAI Pythia family (GPT-NeoX architecture, gpt_neox.layers)
    # Trained on the Pile (pre-2023); use for WikiMIA experiments.
    # ==========================================================================
    "EleutherAI/pythia-1b": ModelSpec(
        name="EleutherAI/pythia-1b",
        d_model=2048,
        n_layers=16,
        n_heads=8,
        vocab_size=50304,
        tokenizer_name="EleutherAI/pythia-1b",
        architecture="gpt_neox",
    ),
    "pythia-1b": ModelSpec(
        name="pythia-1b",
        d_model=2048,
        n_layers=16,
        n_heads=8,
        vocab_size=50304,
        tokenizer_name="EleutherAI/pythia-1b",
        architecture="gpt_neox",
    ),
    "EleutherAI/pythia-1.4b": ModelSpec(
        name="EleutherAI/pythia-1.4b",
        d_model=2048,
        n_layers=24,
        n_heads=16,
        vocab_size=50304,
        tokenizer_name="EleutherAI/pythia-1.4b",
        architecture="gpt_neox",
    ),
    "pythia-1.4b": ModelSpec(
        name="pythia-1.4b",
        d_model=2048,
        n_layers=24,
        n_heads=16,
        vocab_size=50304,
        tokenizer_name="EleutherAI/pythia-1.4b",
        architecture="gpt_neox",
    ),
    "EleutherAI/pythia-6.9b": ModelSpec(
        name="EleutherAI/pythia-6.9b",
        d_model=4096,
        n_layers=32,
        n_heads=32,
        vocab_size=50304,
        tokenizer_name="EleutherAI/pythia-6.9b",
        architecture="gpt_neox",
    ),
    "pythia-6.9b": ModelSpec(
        name="pythia-6.9b",
        d_model=4096,
        n_layers=32,
        n_heads=32,
        vocab_size=50304,
        tokenizer_name="EleutherAI/pythia-6.9b",
        architecture="gpt_neox",
    ),

    # ==========================================================================
    # EleutherAI GPT-Neo family (GPT-2/Neo architecture, transformer.h)
    # Trained on the Pile (pre-2023); use for WikiMIA experiments.
    # ==========================================================================
    "EleutherAI/gpt-neo-1.3B": ModelSpec(
        name="EleutherAI/gpt-neo-1.3B",
        d_model=2048,
        n_layers=24,
        n_heads=16,
        vocab_size=50257,
        tokenizer_name="EleutherAI/gpt-neo-1.3B",
        architecture="gpt2",
    ),
    "gpt-neo-1.3B": ModelSpec(
        name="gpt-neo-1.3B",
        d_model=2048,
        n_layers=24,
        n_heads=16,
        vocab_size=50257,
        tokenizer_name="EleutherAI/gpt-neo-1.3B",
        architecture="gpt2",
    ),
    "EleutherAI/gpt-neo-2.7B": ModelSpec(
        name="EleutherAI/gpt-neo-2.7B",
        d_model=2560,
        n_layers=32,
        n_heads=20,
        vocab_size=50257,
        tokenizer_name="EleutherAI/gpt-neo-2.7B",
        architecture="gpt2",
    ),
    "gpt-neo-2.7B": ModelSpec(
        name="gpt-neo-2.7B",
        d_model=2560,
        n_layers=32,
        n_heads=20,
        vocab_size=50257,
        tokenizer_name="EleutherAI/gpt-neo-2.7B",
        architecture="gpt2",
    ),

    # OLMo family
    "olmo_1B_ppo": ModelSpec(
        name="olmo_1B_ppo",
        d_model=2048,
        n_layers=16,
        n_heads=16,
        vocab_size=32000,
        tokenizer_name="olmo_1B_ppo",
        architecture="olmo",
        rope_theta=10000.0,
    ),

    # Llama-3.2-3B fine-tunes
    "Llama-3.2-3B-GSM8K-ft-3eps": ModelSpec(
        name="Llama-3.2-3B-GSM8K-ft-3eps",
        d_model=3072,
        n_layers=28,
        n_heads=24,
        vocab_size=128256,
        tokenizer_name="Llama-3.2-3B-GSM8K-ft-3eps",
        architecture="llama",
        rope_theta=500000.0,
    ),

    # SmolLM
    "HuggingFaceTB/SmolLM-135M": ModelSpec(
        name="HuggingFaceTB/SmolLM-135M",
        d_model=576,
        n_layers=30,
        n_heads=9,
        vocab_size=49152,
        tokenizer_name="HuggingFaceTB/SmolLM-135M",
        architecture="llama",
        rope_theta=10000.0,
        intermediate_size=1536,
    ),
    "HuggingFaceTB/SmolLM-360M": ModelSpec(
        name="HuggingFaceTB/SmolLM-360M",
        d_model=960,
        n_layers=32,
        n_heads=15,
        vocab_size=49152,
        tokenizer_name="HuggingFaceTB/SmolLM-360M",
        architecture="llama",
        rope_theta=10000.0,
        intermediate_size=2560,
    ),

    # Qwen 3 family
    "Qwen/Qwen3-8B": ModelSpec(
        name="Qwen/Qwen3-8B",
        d_model=4096,
        n_layers=36,
        n_heads=32,
        vocab_size=151936,
        tokenizer_name="Qwen/Qwen3-8B",
        architecture="qwen3",
        rope_theta=1000000.0,
        intermediate_size=12288,
    ),

    # Qwen 3.5 family
    "Qwen/Qwen3.5-4B": ModelSpec(
        name="Qwen/Qwen3.5-4B",
        d_model=2560,
        n_layers=32,
        n_heads=16,
        vocab_size=248320,
        tokenizer_name="Qwen/Qwen3.5-4B",
        architecture="qwen3_5",
        rope_theta=10000000.0,
        intermediate_size=9216,
    ),
    "Qwen/Qwen3.5-9B": ModelSpec(
        name="Qwen/Qwen3.5-9B",
        d_model=4096,
        n_layers=32,
        n_heads=16,
        vocab_size=248320,
        tokenizer_name="Qwen/Qwen3.5-9B",
        architecture="qwen3_5",
        rope_theta=10000000.0,
        intermediate_size=12288,
    ),

    # TinyLlama
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": ModelSpec(
        name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        d_model=2048,
        n_layers=22,
        n_heads=32,
        vocab_size=32000,
        tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        architecture="llama",
        rope_theta=10000.0,
        intermediate_size=5632,
    ),
    "TinyLlama/TinyLlama_v1.1": ModelSpec(
        name="TinyLlama/TinyLlama_v1.1",
        d_model=2048,
        n_layers=22,
        n_heads=32,
        vocab_size=32000,
        tokenizer_name="TinyLlama/TinyLlama_v1.1",
        architecture="llama",
        rope_theta=10000.0,
        intermediate_size=5632,
    ),

    # Phi-2
    "microsoft/phi-2": ModelSpec(
        name="microsoft/phi-2",
        d_model=2560,
        n_layers=32,
        n_heads=32,
        vocab_size=51200,
        tokenizer_name="microsoft/phi-2",
        architecture="phi",
        rope_theta=10000.0,
        intermediate_size=10240,
    ),

    # SmolLM
    "HuggingFaceTB/SmolLM-135M": ModelSpec(
        name="HuggingFaceTB/SmolLM-135M",
        d_model=576,
        n_layers=30,
        n_heads=9,
        vocab_size=49152,
        tokenizer_name="HuggingFaceTB/SmolLM-135M",
        architecture="llama",
        rope_theta=10000.0,
        intermediate_size=1536,
    ),
    "HuggingFaceTB/SmolLM-360M": ModelSpec(
        name="HuggingFaceTB/SmolLM-360M",
        d_model=960,
        n_layers=32,
        n_heads=15,
        vocab_size=49152,
        tokenizer_name="HuggingFaceTB/SmolLM-360M",
        architecture="llama",
        rope_theta=10000.0,
        intermediate_size=2560,
    ),

    # ==========================================================================
    # LLaMA 3 family (used in the paper arXiv:2501.14082)
    # ==========================================================================
    "meta-llama/Llama-3.2-1B": ModelSpec(
        name="meta-llama/Llama-3.2-1B",
        d_model=2048,
        n_layers=16,
        n_heads=32,
        vocab_size=128256,
        tokenizer_name="meta-llama/Llama-3.2-1B",
        architecture="llama",
        rope_theta=500000.0,
        intermediate_size=8192,
    ),
    "meta-llama/Llama-3.2-1B-Instruct": ModelSpec(
        name="meta-llama/Llama-3.2-1B-Instruct",
        d_model=2048,
        n_layers=16,
        n_heads=32,
        vocab_size=128256,
        tokenizer_name="meta-llama/Llama-3.2-1B-Instruct",
        architecture="llama",
        rope_theta=500000.0,
        intermediate_size=8192,
    ),
    "meta-llama/Llama-3.2-3B": ModelSpec(
        name="meta-llama/Llama-3.2-3B",
        d_model=3072,
        n_layers=28,
        n_heads=24,
        vocab_size=128256,
        tokenizer_name="meta-llama/Llama-3.2-3B",
        architecture="llama",
        rope_theta=500000.0,
        intermediate_size=8192,
    ),
    "meta-llama/Llama-3.2-3B-Instruct": ModelSpec(
        name="meta-llama/Llama-3.2-3B-Instruct",
        d_model=3072,
        n_layers=28,
        n_heads=24,
        vocab_size=128256,
        tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",
        architecture="llama",
        rope_theta=500000.0,
        intermediate_size=8192,
    ),
    "Hermes-3-Llama-3.2-3B": ModelSpec(
        name="NousResearch/Hermes-3-Llama-3.2-3B",
        d_model=3072,
        n_layers=28,
        n_heads=24,
        vocab_size=128256,
        tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",
        architecture="llama",
        rope_theta=500000.0,
        intermediate_size=8192,
    ),
    "Llama-Deepsync-3B": ModelSpec(
        name="NousResearch/Llama-Deepsync-3B",
        d_model=3072,
        n_layers=28,
        n_heads=24,
        vocab_size=128256,
        tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",
        architecture="llama",
        rope_theta=500000.0,
        intermediate_size=8192,
    ),
    "Primal-Mini-3B-Exp": ModelSpec(
        name="Primal-Mini-3B-Exp",
        d_model=3072,
        n_layers=28,
        n_heads=24,
        vocab_size=128256,
        tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",
        architecture="llama",
        rope_theta=500000.0,
        intermediate_size=8192,
    ),
    "thea-3b-25r": ModelSpec(
        name="thea-3b-25r",
        d_model=3072,
        n_layers=28,
        n_heads=24,
        vocab_size=128256,
        tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",
        architecture="llama",
        rope_theta=500000.0,
        intermediate_size=8192,
    ),
    "kmseong_Llama3.2-3B-gsm8k-fullft": ModelSpec(
        name="kmseong_Llama3.2-3B-gsm8k-fullft",
        d_model=3072,
        n_layers=28,
        n_heads=24,
        vocab_size=128256,
        tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",
        architecture="llama",
        rope_theta=500000.0,
        intermediate_size=8192,
    ),
    "colesmcintosh_llama-3.2-3B-gsm8k": ModelSpec(
        name="colesmcintosh_llama-3.2-3B-gsm8k",
        d_model=3072,
        n_layers=28,
        n_heads=24,
        vocab_size=128256,
        tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",
        architecture="llama",
        rope_theta=500000.0,
        intermediate_size=8192,
    ),
    "meta-llama/Llama-3.1-8B": ModelSpec(
        name="meta-llama/Llama-3.1-8B",
        d_model=4096,
        n_layers=32,
        n_heads=32,
        vocab_size=128256,
        tokenizer_name="meta-llama/Llama-3.1-8B",
        architecture="llama",
        rope_theta=500000.0,
        intermediate_size=14336,
    ),
    "meta-llama/Llama-3.1-8B-Instruct": ModelSpec(
        name="meta-llama/Llama-3.1-8B-Instruct",
        d_model=4096,
        n_layers=32,
        n_heads=32,
        vocab_size=128256,
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        architecture="llama",
        rope_theta=500000.0,
        intermediate_size=14336,
    ),

    # Mistral
    "Mistral-Nemo-Instruct-2407": ModelSpec(
        name="Mistral-Nemo-Instruct-2407",
        d_model=5120,
        n_layers=40,
        n_heads=32,
        vocab_size=131072,
        tokenizer_name="Mistral-Nemo-Instruct-2407",
        architecture="mistral",
        rope_theta=1000000.0,
        intermediate_size=14336,
    ),
    "mistralai/Mistral-Nemo-Instruct-2407": ModelSpec(
        name="mistralai/Mistral-Nemo-Instruct-2407",
        d_model=5120,
        n_layers=40,
        n_heads=32,
        vocab_size=131072,
        tokenizer_name="mistralai/Mistral-Nemo-Instruct-2407",
        architecture="mistral",
        rope_theta=1000000.0,
        intermediate_size=14336,
    ),
}


def get_model_spec(model_name: str) -> ModelSpec:
    """
    Get specification for a model.

    Args:
        model_name: HuggingFace model name

    Returns:
        ModelSpec with model metadata

    Raises:
        ValueError: If model not in registry
    """
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    # Fall back: match by basename for local paths
    import os  # pylint: disable=import-outside-toplevel
    basename = os.path.basename(model_name.rstrip("/"))
    for key, spec in MODEL_REGISTRY.items():
        if key.split("/")[-1] == basename:
            return spec

    available = list(MODEL_REGISTRY.keys())
    raise ValueError(
        f"Model '{model_name}' not in registry. "
        f"Available models: {available}. "
        f"Add it to MODEL_REGISTRY in model_registry.py"
    )


def check_dimension_compatibility(model_a: str, model_b: str) -> Dict[str, Any]:
    """
    Check dimension compatibility between two models.

    Args:
        model_a: Name of model A (sender)
        model_b: Name of model B (receiver)

    Returns:
        Dictionary with compatibility information
    """
    spec_a = get_model_spec(model_a)
    spec_b = get_model_spec(model_b)

    return {
        "d_model_a": spec_a.d_model,
        "d_model_b": spec_b.d_model,
        "d_model_match": spec_a.d_model == spec_b.d_model,
        "requires_projection": spec_a.d_model != spec_b.d_model,
        "projection_shape": (
            (spec_b.d_model, spec_a.d_model)
            if spec_a.d_model != spec_b.d_model
            else None
        ),
        "n_layers_a": spec_a.n_layers,
        "n_layers_b": spec_b.n_layers,
        "same_architecture": spec_a.architecture == spec_b.architecture,
        "same_tokenizer": spec_a.tokenizer_name == spec_b.tokenizer_name,
        "architecture_a": spec_a.architecture,
        "architecture_b": spec_b.architecture,
    }


def register_model(spec: ModelSpec) -> None:
    """
    Register a new model specification.

    Args:
        spec: ModelSpec to register
    """
    MODEL_REGISTRY[spec.name] = spec


def list_models() -> Dict[str, Dict[str, Any]]:
    """
    List all registered models with key info.

    Returns:
        Dictionary mapping model names to their key properties
    """
    return {
        name: {
            "d_model": spec.d_model,
            "n_layers": spec.n_layers,
            "architecture": spec.architecture,
        }
        for name, spec in MODEL_REGISTRY.items()
    }


if __name__ == "__main__":
    # Print registry info
    print("Registered Models:")
    print("=" * 80)
    for name, info in list_models().items():
        print(f"  {name}")
        print(
            f"    d_model={info['d_model']}, layers={info['n_layers']}, "
            f"arch={info['architecture']}"
        )

    print("\n\nCompatibility Check Example:")
    print("=" * 80)
    compat = check_dimension_compatibility("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B")
    for k, v in compat.items():
        print(f"  {k}: {v}")
