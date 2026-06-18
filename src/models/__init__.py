"""Model loading and activation extraction."""

from src.models.model_registry import (
    ModelSpec,
    MODEL_REGISTRY,
    get_model_spec,
    check_dimension_compatibility,
)
from src.models.activation_extractor import (
    ExtractionPoint,
    ExtractedActivation,
    ActivationExtractor,
    ActivationInjector,
)
from src.models.layer_utils import (
    get_transformer_layers,
    unpack_hook_output,
    repack_hook_output,
)
from src.models.model_loader import load_model_and_tokenizer
from src.models.inference_utils import (
    build_chat_prompt,
    generate_response,
    tokenize_and_generate,
)

__all__ = [
    "ModelSpec",
    "MODEL_REGISTRY",
    "get_model_spec",
    "check_dimension_compatibility",
    "ExtractionPoint",
    "ExtractedActivation",
    "ActivationExtractor",
    "ActivationInjector",
    "get_transformer_layers",
    "unpack_hook_output",
    "repack_hook_output",
    "load_model_and_tokenizer",
    "build_chat_prompt",
    "generate_response",
    "tokenize_and_generate",
]
