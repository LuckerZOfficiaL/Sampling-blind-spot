"""Hook-based activation extraction from transformer models."""

from enum import Enum
from typing import Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
import torch
from torch import Tensor
from torch import nn
from transformers import PreTrainedModel

from src.models.layer_utils import (
    get_transformer_layers,
    unpack_hook_output,
    repack_hook_output,
)


class ExtractionPoint(Enum):
    """
    Where to extract activations within a transformer layer.

    For a standard pre-norm transformer layer:
        Input
          → LayerNorm → Attention → + (residual)  -> POST_ATTENTION
          → LayerNorm → MLP → + (residual)        -> POST_MLP (residual stream)

    POST_MLP is the standard "residual stream" used in most interpretability work.
    """

    PRE_ATTENTION = "pre_attention"  # After input LayerNorm, before attention
    POST_ATTENTION = "post_attention"  # After attention + residual
    PRE_MLP = "pre_mlp"  # After post-attention LayerNorm, before MLP
    POST_MLP = "post_mlp"  # After MLP + residual (standard residual stream)


@dataclass
class ExtractedActivation:
    """Container for extracted activation with metadata."""

    tensor: Tensor  # Shape: [batch, d_model]
    layer_idx: int
    extraction_point: ExtractionPoint
    token_position: int  # Which token position was extracted (-1 = last)
    sequence_length: int  # Original sequence length
    model_name: str


class ActivationExtractor:
    """
    Extract intermediate activations from transformer layers using hooks.

    Supports extracting from multiple layers simultaneously and handles
    different model architectures (Llama, Qwen, Phi, GPT-2, etc.).

    Usage:
        extractor = ActivationExtractor(model, layer_indices=[10, 15, 20])

        # Context manager usage
        with extractor:
            outputs = model(**inputs)
            activations = extractor.get_activations()

        # Or direct extraction
        activations = extractor.extract(input_ids, attention_mask)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        layer_indices: List[int],
        extraction_point: ExtractionPoint = ExtractionPoint.POST_MLP,
        token_position: int = -1,  # -1 = final token
    ):
        """
        Initialize the activation extractor.

        Args:
            model: HuggingFace model to extract from
            layer_indices: List of layer indices to extract from
            extraction_point: Where in each layer to extract
            token_position: Which token position to extract (-1 for last)
        """
        self.model = model
        self.layer_indices = sorted(layer_indices)
        self.extraction_point = extraction_point
        self.token_position = token_position

        self._activations: Dict[int, Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._model_name = getattr(model.config, "_name_or_path", "unknown")

        # Validate layer indices
        n_layers = self._get_num_layers()
        for idx in layer_indices:
            if idx < 0 or idx >= n_layers:
                raise ValueError(
                    f"Layer index {idx} out of range [0, {n_layers})"
                )

        # Find layer modules
        self._layer_modules = self._find_layer_modules()

    def _get_num_layers(self) -> int:
        """Get the number of layers in the model."""
        return getattr(self.model.config, "num_hidden_layers", 0)

    def _get_layers(self) -> nn.ModuleList:
        """Get the transformer layers from the model."""
        return get_transformer_layers(self.model)

    def _find_layer_modules(self) -> Dict[int, nn.Module]:  # pylint: disable=too-many-branches
        """
        Find the appropriate module for each layer based on extraction point.
        """
        layers = self._get_layers()
        modules = {}

        for idx in self.layer_indices:
            layer = layers[idx]

            if self.extraction_point == ExtractionPoint.POST_MLP:
                modules[idx] = layer

            elif self.extraction_point == ExtractionPoint.POST_ATTENTION:
                if hasattr(layer, "self_attn"):
                    modules[idx] = layer.self_attn
                elif hasattr(layer, "attention"):
                    modules[idx] = layer.attention
                elif hasattr(layer, "attn"):
                    modules[idx] = layer.attn
                else:
                    raise ValueError(
                        f"Cannot find attention module in layer {idx}"
                    )

            elif self.extraction_point == ExtractionPoint.PRE_MLP:
                if hasattr(layer, "post_attention_layernorm"):
                    modules[idx] = layer.post_attention_layernorm
                elif hasattr(layer, "ln_2"):
                    modules[idx] = layer.ln_2
                else:
                    raise ValueError(
                        f"Cannot find pre-MLP layernorm in layer {idx}"
                    )

            elif self.extraction_point == ExtractionPoint.PRE_ATTENTION:
                if hasattr(layer, "input_layernorm"):
                    modules[idx] = layer.input_layernorm
                elif hasattr(layer, "ln_1"):
                    modules[idx] = layer.ln_1
                else:
                    raise ValueError(
                        f"Cannot find input layernorm in layer {idx}"
                    )

        return modules

    def _make_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook that captures activations."""

        def hook(
            _module: nn.Module,
            _input: Union[Tensor, Tuple[Tensor, ...]],
            output: Union[Tensor, Tuple[Tensor, ...]],
        ):
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Ensure we have a 3D tensor [batch, seq_len, d_model]
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

            # Determine token position
            seq_len = hidden_states.shape[1]
            pos = self.token_position
            if pos < 0:
                pos = seq_len + pos  # Convert negative index

            if pos < 0 or pos >= seq_len:
                raise ValueError(
                    f"Token position {self.token_position} out of range for "
                    f"sequence length {seq_len}"
                )

            # Extract activation at specified position
            # Shape: [batch, d_model]
            activation = hidden_states[:, pos, :].detach().clone()
            self._activations[layer_idx] = activation

        return hook

    def _register_hooks(self) -> None:
        """Register forward hooks on target modules."""
        for layer_idx, module in self._layer_modules.items():
            hook = module.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self) -> "ActivationExtractor":
        """Context manager entry - register hooks."""
        self._activations.clear()
        self._register_hooks()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit - remove hooks."""
        self._remove_hooks()

    def get_activations(self) -> Dict[int, Tensor]:
        """
        Get extracted activations after forward pass.

        Returns:
            Dict mapping layer_idx -> activation tensor [batch, d_model]
        """
        return self._activations.copy()

    def get_activation(self, layer_idx: int) -> Optional[Tensor]:
        """
        Get activation for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Activation tensor or None if not extracted
        """
        return self._activations.get(layer_idx)

    def extract(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[int, ExtractedActivation]:
        """
        Run extraction in one call.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Dict mapping layer_idx to ExtractedActivation dataclass
        """
        seq_len = input_ids.shape[1]

        with self:
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask)

        results = {}
        for layer_idx, tensor in self._activations.items():
            pos = self.token_position
            if pos < 0:
                pos = seq_len + pos

            results[layer_idx] = ExtractedActivation(
                tensor=tensor,
                layer_idx=layer_idx,
                extraction_point=self.extraction_point,
                token_position=pos,
                sequence_length=seq_len,
                model_name=self._model_name,
            )

        return results

    def extract_all_layers(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[int, Tensor]:
        """
        Extract activations from all layers (utility method).

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask

        Returns:
            Dict mapping layer_idx -> activation tensor for all layers
        """
        n_layers = self._get_num_layers()
        all_layers = list(range(n_layers))

        # Create temporary extractor for all layers
        extractor = ActivationExtractor(
            self.model,
            all_layers,
            self.extraction_point,
            self.token_position,
        )

        with extractor:
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask)
            return extractor.get_activations()


class ActivationInjector:
    """
    Inject activations into a model's forward pass.

    Used to implement activation grafting by replacing a model's
    intermediate activation with a combined activation.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        layer_idx: int,
        extraction_point: ExtractionPoint = ExtractionPoint.POST_MLP,
    ):
        """
        Initialize the activation injector.

        Args:
            model: Model to inject into
            layer_idx: Layer at which to inject
            extraction_point: Where in the layer to inject
        """
        self.model = model
        self.layer_idx = layer_idx
        self.extraction_point = extraction_point

        self._injection_activation: Optional[Tensor] = None
        self._injection_position: int = -1
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Find target module
        self._target_module = self._find_target_module()

    def _get_layers(self) -> nn.ModuleList:
        """Get transformer layers from the model."""
        return get_transformer_layers(self.model)

    def _find_target_module(self) -> nn.Module:  # pylint: disable=too-many-return-statements
        """Find the module to hook for injection."""
        layers = self._get_layers()
        layer = layers[self.layer_idx]

        if self.extraction_point == ExtractionPoint.POST_MLP:
            return layer
        if self.extraction_point == ExtractionPoint.POST_ATTENTION:
            if hasattr(layer, "self_attn"):
                return layer.self_attn
            if hasattr(layer, "attn"):
                return layer.attn
            raise ValueError("Cannot find attention module")
        if self.extraction_point == ExtractionPoint.PRE_MLP:
            if hasattr(layer, "post_attention_layernorm"):
                return layer.post_attention_layernorm
            if hasattr(layer, "ln_2"):
                return layer.ln_2
            raise ValueError("Cannot find pre-MLP layernorm")
        if self.extraction_point == ExtractionPoint.PRE_ATTENTION:
            if hasattr(layer, "input_layernorm"):
                return layer.input_layernorm
            if hasattr(layer, "ln_1"):
                return layer.ln_1
            raise ValueError("Cannot find input layernorm")
        raise ValueError(
            f"Unknown extraction point: {self.extraction_point}"
        )

    def _make_injection_hook(self) -> Callable:
        """Create hook that performs activation injection."""

        def hook(
            _module: nn.Module,
            _input: Union[Tensor, Tuple[Tensor, ...]],
            output: Union[Tensor, Tuple[Tensor, ...]],
        ):
            if self._injection_activation is None:
                return output

            hidden_states, rest = unpack_hook_output(output)

            # Determine position
            seq_len = hidden_states.shape[1]
            pos = self._injection_position
            if pos < 0:
                pos = seq_len + pos

            # Clone to avoid in-place modification issues
            hidden_states = hidden_states.clone()

            # Inject activation
            hidden_states[:, pos, :] = self._injection_activation

            return repack_hook_output(hidden_states, rest)

        return hook

    def set_injection(
        self,
        activation: Tensor,
        position: int = -1,
    ) -> None:
        """
        Set the activation to inject.

        Args:
            activation: Activation tensor [batch, d_model]
            position: Token position at which to inject (-1 for last)
        """
        self._injection_activation = activation
        self._injection_position = position

    def clear_injection(self) -> None:
        """Clear the injection activation."""
        self._injection_activation = None

    def _register_hooks(self) -> None:
        """Register injection hook."""
        hook = self._target_module.register_forward_hook(self._make_injection_hook())
        self._hooks.append(hook)

    def _remove_hooks(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self) -> "ActivationInjector":
        """Context manager entry."""
        self._register_hooks()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self._remove_hooks()
        self.clear_injection()


if __name__ == "__main__":
    # Test extraction
    print("Testing activation extraction...")

    # pylint: disable=ungrouped-imports
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TEST_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {TEST_MODEL_NAME}...")

    test_model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    test_tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME, trust_remote_code=True)

    # Test input
    TEST_TEXT = "The capital of France is"
    inputs = test_tokenizer(TEST_TEXT, return_tensors="pt").to(test_model.device)

    print(f"\nInput: '{TEST_TEXT}'")
    print(f"Input shape: {inputs['input_ids'].shape}")

    # Test extraction from multiple layers
    test_extractor = ActivationExtractor(
        test_model,
        layer_indices=[0, 12, 23],
        extraction_point=ExtractionPoint.POST_MLP,
        token_position=-1,
    )

    activations = test_extractor.extract(inputs["input_ids"], inputs.get("attention_mask"))

    print("\nExtracted activations:")
    for test_layer_idx, act in activations.items():
        print(
            f"  Layer {test_layer_idx}: shape={act.tensor.shape},"
            f" norm={act.tensor.norm().item():.4f}"
        )

    print("\nTest passed!")
