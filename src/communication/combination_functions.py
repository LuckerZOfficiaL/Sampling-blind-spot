"""Combination functions for merging activations from two models."""
# pylint: disable=invalid-name  # Math notation: W, Q, K_a, K_b

from abc import ABC, abstractmethod
from typing import Optional, Iterator, Dict, Any
import torch
from torch import Tensor
from torch import nn


class CombinationFunction(ABC):
    """
    Abstract base class for activation combination strategies.

    A combination function takes activations from Model A (sender) and
    Model B (receiver) and produces a combined activation to inject
    into Model B's forward pass.
    """

    @abstractmethod
    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Combine activations from Model A and Model B.

        Args:
            a: Activation from Model A, shape [batch, d_a]
            b: Activation from Model B, shape [batch, d_b]

        Returns:
            Combined activation, shape [batch, d_b]
        """

    @property
    def trainable(self) -> bool:
        """Whether this combination function has trainable parameters."""
        return False

    def parameters(self) -> Iterator[nn.Parameter]:
        """Return trainable parameters (empty by default)."""
        return iter([])

    def to(self, device: torch.device) -> "CombinationFunction":  # pylint: disable=unused-argument
        """Move to device (no-op for non-trainable functions)."""
        return self

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for serialization."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:  # noqa: B027
        """Load state dict."""

    def combine_with_context(self, a: Tensor, hidden_states: Tensor, pos: int) -> Tensor:
        """
        Combine activations with access to the full hidden state tensor.

        Default implementation extracts hidden_states[:, pos, :] and calls combine().
        Override this method for strategies that need access to other token positions
        (e.g., AverageActivationCombination).

        Args:
            a: Activation from Model A, shape [batch, d_a]
            hidden_states: Full hidden state tensor from Model B, shape [batch, seq_len, d_b]
            pos: Graft position (non-negative index)

        Returns:
            Combined activation, shape [batch, d_b]
        """
        return self.combine(a, hidden_states[:, pos, :])


class SumCombination(CombinationFunction):
    """
    f(a, b) = a + b

    Simple additive combination. Requires d_a == d_b.
    """

    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        if a.shape[-1] != b.shape[-1]:
            raise ValueError(
                f"SumCombination requires matching dimensions: "
                f"d_a={a.shape[-1]}, d_b={b.shape[-1]}"
            )
        return a + b


class MeanCombination(CombinationFunction):
    """
    f(a, b) = 0.5 * (a + b)

    Averaged combination. Requires d_a == d_b.
    """

    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        if a.shape[-1] != b.shape[-1]:
            raise ValueError(
                f"MeanCombination requires matching dimensions: "
                f"d_a={a.shape[-1]}, d_b={b.shape[-1]}"
            )
        return 0.5 * (a + b)


class ReplaceCombination(CombinationFunction):
    """
    f(a, b) = a (or project(a) if dimensions differ)

    Replaces B's activation entirely with A's.
    If dimensions differ, uses a random orthogonal projection that
    preserves norms and angles (Johnson-Lindenstrauss lemma).
    """

    def __init__(self, d_a: Optional[int] = None, d_b: Optional[int] = None, seed: int = 42):
        """
        Initialize ReplaceCombination.

        Args:
            d_a: Dimension of Model A's activations (for projection)
            d_b: Dimension of Model B's activations (for projection)
            seed: Random seed for reproducible orthogonal projection
        """
        self._orthogonal_proj: Optional[Tensor] = None
        self.d_a = d_a
        self.d_b = d_b

        if d_a is not None and d_b is not None and d_a != d_b:
            # Create random orthogonal projection matrix
            self._orthogonal_proj = self._create_orthogonal_projection(d_a, d_b, seed)

    def _create_orthogonal_projection(self, d_a: int, d_b: int, seed: int) -> Tensor:
        """
        Create a random orthogonal projection matrix from d_a to d_b.

        Uses QR decomposition of a random Gaussian matrix to get
        orthonormal columns, preserving norms and angles per
        Johnson-Lindenstrauss lemma.
        """
        generator = torch.Generator().manual_seed(seed)

        if d_a < d_b:
            # Expanding: d_a -> d_b
            # Create random matrix and orthogonalize
            random_matrix = torch.randn(d_b, d_a, generator=generator)
            Q, _ = torch.linalg.qr(random_matrix)
            # Q is (d_b, d_a) with orthonormal columns
            # Projection: a @ Q.T gives (batch, d_b)
            return Q  # Shape: (d_b, d_a)

        # Contracting: d_a -> d_b
        # Create random matrix and orthogonalize
        random_matrix = torch.randn(d_a, d_b, generator=generator)
        Q, _ = torch.linalg.qr(random_matrix)
        # Q is (d_a, d_b) with orthonormal columns
        # Projection: a @ Q gives (batch, d_b)
        return Q  # Shape: (d_a, d_b)

    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        d_a, d_b = a.shape[-1], b.shape[-1]
        if d_a == d_b:
            return a

        # Use orthogonal projection
        if self._orthogonal_proj is None:
            # Lazily create projection if not initialized with dimensions
            self._orthogonal_proj = self._create_orthogonal_projection(d_a, d_b, seed=42)

        proj = self._orthogonal_proj.to(a.device, a.dtype)

        if d_a < d_b:
            # Expanding: proj is (d_b, d_a), compute a @ proj.T
            return a @ proj.T

        # Contracting: proj is (d_a, d_b), compute a @ proj
        return a @ proj

    def to(self, device: torch.device) -> "ReplaceCombination":
        if self._orthogonal_proj is not None:
            self._orthogonal_proj = self._orthogonal_proj.to(device)
        return self


class TrainedProjectionCombination(CombinationFunction):
    """
    f(a, b) = W @ a

    Uses a pre-trained projection matrix W to map activations from
    Model A's space to Model B's space. W is trained on C4 sentences
    to minimize MSE between projected A activations and B activations.
    """

    def __init__(self, projection_path: str, normalize_source: bool = False):
        """
        Initialize with a pre-trained projection matrix.

        Args:
            projection_path: Path to saved projection (.pt file)
            normalize_source: If True, L2-normalize the source activation before
                applying W. Required when W was trained with normalized_mse loss.
        """
        checkpoint = torch.load(projection_path, map_location="cpu")
        self._weight = checkpoint["weight"]  # Shape: (d_b, d_a)
        self.d_a = checkpoint["d_a"]
        self.d_b = checkpoint["d_b"]
        self.normalize_source = normalize_source
        self.metadata = {
            "layer_a": checkpoint.get("layer_a"),
            "layer_b": checkpoint.get("layer_b"),
            "model_a": checkpoint.get("model_a"),
            "model_b": checkpoint.get("model_b"),
            "epochs": checkpoint.get("epochs"),
            "n_sentences": checkpoint.get("n_sentences"),
        }

    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        d_a, d_b = a.shape[-1], b.shape[-1]
        if d_a == d_b:
            return a

        if self.normalize_source:
            a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Apply trained projection: W @ a
        # W is (d_b, d_a), a is (..., d_a), result is (..., d_b)
        W = self._weight.to(a.device, a.dtype)
        return a @ W.T

    def to(self, device: torch.device) -> "TrainedProjectionCombination":
        self._weight = self._weight.to(device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        return {"weight": self._weight.clone(), **self.metadata}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "weight" in state_dict:
            self._weight = state_dict["weight"]


class RandomNoiseCombination(CombinationFunction):
    """
    f(a, b) = randn_like(b) * std

    Ignores a entirely. Returns Gaussian noise with the same shape and device
    as b (model B's activation). Used as a null/control baseline.
    """

    def __init__(self, std: float = 0.01):
        self.std = std

    def combine(self, a, b: Tensor) -> Tensor:
        return torch.randn_like(b) * self.std

    def to(self, device) -> "RandomNoiseCombination":
        return self  # no parameters to move

    def state_dict(self) -> Dict[str, Any]:
        return {"std": self.std}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "std" in state_dict:
            self.std = state_dict["std"]


class RandomAddCombination(CombinationFunction):
    """
    f(a, b) = b + randn_like(b) * std

    Ignores a entirely. Adds Gaussian noise to b (target activation).
    Used as a null/control baseline for the "sum" grafting mode.
    """

    def __init__(self, std: float = 0.01):
        self.std = std

    def combine(self, a, b: Tensor) -> Tensor:
        return b + torch.randn_like(b) * self.std

    def to(self, device) -> "RandomAddCombination":
        return self

    def state_dict(self) -> Dict[str, Any]:
        return {"std": self.std}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "std" in state_dict:
            self.std = state_dict["std"]


class ZeroVectorCombination(CombinationFunction):
    """
    f(a, b) = zeros_like(b)

    Ignores a entirely. Returns a zero vector with the same shape and device
    as b (model B's activation). Used as a null/control baseline.
    """

    def combine(self, a, b: Tensor) -> Tensor:
        return torch.zeros_like(b)

    def to(self, device) -> "ZeroVectorCombination":
        return self  # no parameters to move


class AverageActivationCombination(CombinationFunction):
    """
    f(hidden_states, pos) = mean(hidden_states[:, all positions except pos, :], dim=1)

    Replaces the graft-position activation with the mean of all other token positions
    in Model B's hidden state at the graft layer. Ignores Model A entirely.
    Used as a context-average baseline: tests whether injecting the "average context
    representation" disrupts or improves generation relative to zero/random vectors.

    Requires combine_with_context() — combine() raises NotImplementedError.
    """

    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError(
            "AverageActivationCombination requires the full hidden state. "
            "Use combine_with_context() instead."
        )

    def combine_with_context(self, a: Tensor, hidden_states: Tensor, pos: int) -> Tensor:
        seq_len = hidden_states.shape[1]
        if seq_len <= 1:
            # Only one token — fall back to zeros
            return torch.zeros_like(hidden_states[:, pos, :])
        # Gather all positions except pos
        indices = [i for i in range(seq_len) if i != pos]
        other = hidden_states[:, indices, :]  # [batch, seq_len-1, d]
        return other.mean(dim=1)

    def to(self, device) -> "AverageActivationCombination":
        return self


class RandomUnitCombination(CombinationFunction):
    """
    f(a, b) = randn_like(b) / ||randn_like(b)|| * ||b||

    Random direction normalized to match the L2 norm of b (model B's activation).
    Controls for the magnitude difference between random/zero vectors and typical
    activations: same norm as the real activation, but random direction.
    Ignores a entirely.
    """

    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        noise = torch.randn_like(b)
        noise_norm = noise.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        b_norm = b.norm(dim=-1, keepdim=True)
        return noise / noise_norm * b_norm

    def to(self, device) -> "RandomUnitCombination":
        return self


class ShuffledActivationCombination(CombinationFunction):
    """
    f(a, b) = shuffle(b along hidden dim)

    Takes b (model B's last-token activation at the graft layer) and randomly
    permutes its values along the hidden dimension. Preserves the marginal value
    distribution and norm of b exactly, but destroys spatial structure.
    Ignores a entirely. A different permutation is drawn for each call.
    """

    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        # b: [batch, d] — apply a single random permutation across the hidden dim
        # (same permutation for all batch items; fine for batch_size=1 inference)
        perm = torch.randperm(b.shape[-1], device=b.device)
        return b[:, perm]

    def to(self, device) -> "ShuffledActivationCombination":
        return self


class BosTokenCombination(CombinationFunction):
    """
    f(hidden_states, pos) = hidden_states[:, 0, :]

    Replaces the graft-position activation with the BOS (first) token's hidden
    state at the same layer. Tests the "reset to beginning of context" signal.
    Ignores a entirely. Requires combine_with_context().
    """

    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError(
            "BosTokenCombination requires the full hidden state. "
            "Use combine_with_context() instead."
        )

    def combine_with_context(self, a: Tensor, hidden_states: Tensor, pos: int) -> Tensor:
        return hidden_states[:, 0, :].clone()

    def to(self, device) -> "BosTokenCombination":
        return self


class PrevLayerActivationCombination(CombinationFunction):
    """
    f(hidden_states_prev, pos) = hidden_states_prev[:, pos, :]

    Replaces the last-token activation at layer_b with the last-token activation
    from layer_b - 1 (one layer below), taken from model B itself.
    Tests whether feeding the "slightly earlier" representation at the same
    position disrupts the wrong reasoning attractor.

    Requires a capture hook registered on layer_b - 1 before generation.
    Call make_capture_hook(graft_position) and register it on the previous layer.
    The ActivationGraftingEngine.generate() detects this automatically.
    """

    def __init__(self) -> None:
        self._captured: Optional[Tensor] = None

    def make_capture_hook(self, graft_position: int):
        """Return a forward hook to register on layer_b - 1."""
        buf = self  # capture self by reference

        def hook(_module, _input, output):
            # output is either a Tensor or a tuple whose first element is the hidden states
            hidden_states = output if isinstance(output, Tensor) else output[0]
            seq_len = hidden_states.shape[1]
            pos = graft_position if graft_position >= 0 else seq_len + graft_position
            pos = max(0, min(pos, seq_len - 1))
            buf._captured = hidden_states[:, pos, :].detach().clone()

        return hook

    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError(
            "PrevLayerActivationCombination requires a capture hook on layer_b-1. "
            "Use combine_with_context() after make_capture_hook() has fired."
        )

    def combine_with_context(self, a: Tensor, hidden_states: Tensor, pos: int) -> Tensor:
        if self._captured is None:
            # Capture hook hasn't fired yet — fall back to zeros
            return torch.zeros_like(hidden_states[:, pos, :])
        return self._captured

    def to(self, device) -> "PrevLayerActivationCombination":
        if self._captured is not None:
            self._captured = self._captured.to(device)
        return self


class ProjectedSumCombination(CombinationFunction):
    """
    f(a, b) = (W @ a) + b

    Projects a into Model B's space using a pre-trained projection matrix W,
    then adds the result to b as a residual. Unlike TrainedProjectionCombination
    (which replaces b entirely) this preserves b's original information.
    """

    def __init__(self, projection_path: str, normalize_source: bool = False):
        checkpoint = torch.load(projection_path, map_location="cpu")
        self._weight = checkpoint["weight"]  # Shape: (d_b, d_a)
        self.d_a = checkpoint["d_a"]
        self.d_b = checkpoint["d_b"]
        self.normalize_source = normalize_source
        self.metadata = {
            "layer_a": checkpoint.get("layer_a"),
            "layer_b": checkpoint.get("layer_b"),
            "model_a": checkpoint.get("model_a"),
            "model_b": checkpoint.get("model_b"),
            "epochs": checkpoint.get("epochs"),
            "n_sentences": checkpoint.get("n_sentences"),
        }

    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        if self.normalize_source:
            a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        W = self._weight.to(a.device, a.dtype)
        return a @ W.T + b  # (..., d_b)

    def to(self, device: torch.device) -> "ProjectedSumCombination":
        self._weight = self._weight.to(device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        return {"weight": self._weight.clone(), **self.metadata}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "weight" in state_dict:
            self._weight = state_dict["weight"]


class ProjectedAverageCombination(CombinationFunction):
    """
    f(a, b) = alpha * (W @ a) + (1 - alpha) * b

    Projects a into Model B's space using a pre-trained projection matrix W,
    then blends with b using a fixed scalar alpha (default 0.5).
    alpha=1.0 is equivalent to trained_projection; alpha=0.5 gives equal weight.
    """

    def __init__(self, projection_path: str, alpha: float = 0.5):
        checkpoint = torch.load(projection_path, map_location="cpu")
        self._weight = checkpoint["weight"]  # Shape: (d_b, d_a)
        self.d_a = checkpoint["d_a"]
        self.d_b = checkpoint["d_b"]
        self.alpha = alpha
        self.metadata = {
            "layer_a": checkpoint.get("layer_a"),
            "layer_b": checkpoint.get("layer_b"),
            "model_a": checkpoint.get("model_a"),
            "model_b": checkpoint.get("model_b"),
            "epochs": checkpoint.get("epochs"),
            "n_sentences": checkpoint.get("n_sentences"),
        }

    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        W = self._weight.to(a.device, a.dtype)
        projected = a @ W.T  # (..., d_b)
        return self.alpha * projected + (1 - self.alpha) * b

    def to(self, device: torch.device) -> "ProjectedAverageCombination":
        self._weight = self._weight.to(device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        return {"weight": self._weight.clone(), "alpha": self.alpha, **self.metadata}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "weight" in state_dict:
            self._weight = state_dict["weight"]
        if "alpha" in state_dict:
            self.alpha = state_dict["alpha"]


class WeightedSumCombination(CombinationFunction):
    """
    f(a, b) = sigmoid(alpha) * a + (1 - sigmoid(alpha)) * b

    Learnable weighted average with a single scalar parameter.
    Requires d_a == d_b.
    """

    def __init__(self, init_alpha: float = 0.0):
        """
        Initialize with alpha (logit-space, so 0 = equal weighting).

        Args:
            init_alpha: Initial value for alpha (in logit space)
        """
        self._alpha = nn.Parameter(torch.tensor(init_alpha))

    @property
    def trainable(self) -> bool:
        return True

    @property
    def alpha(self) -> float:
        """Current alpha value (in probability space)."""
        return torch.sigmoid(self._alpha).item()

    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        if a.shape[-1] != b.shape[-1]:
            raise ValueError(
                f"WeightedSumCombination requires matching dimensions: "
                f"d_a={a.shape[-1]}, d_b={b.shape[-1]}"
            )
        alpha = torch.sigmoid(self._alpha)
        return alpha * a + (1 - alpha) * b

    def parameters(self) -> Iterator[nn.Parameter]:
        return iter([self._alpha])

    def to(self, device: torch.device) -> "WeightedSumCombination":
        self._alpha = nn.Parameter(self._alpha.to(device))
        return self

    def state_dict(self) -> Dict[str, Any]:
        return {"alpha": self._alpha.data.clone()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "alpha" in state_dict:
            self._alpha.data = state_dict["alpha"]


class LearnedLinearCombination(CombinationFunction):
    """
    f(a, b) = W @ a + b

    Projects A's activation into B's space and adds as residual.
    W is trained to minimize alignment loss on held-out data.
    """

    def __init__(self, d_a: int, d_b: int, bias: bool = False):
        """
        Initialize the learned linear combination.

        Args:
            d_a: Dimension of Model A's activations
            d_b: Dimension of Model B's activations
            bias: Whether to include bias in projection
        """
        self.d_a = d_a
        self.d_b = d_b
        self.projection = nn.Linear(d_a, d_b, bias=bias)
        # Initialize with small values for stable training
        nn.init.normal_(self.projection.weight, std=0.02 / (d_a ** 0.5))
        if bias:
            nn.init.zeros_(self.projection.bias)

    @property
    def trainable(self) -> bool:
        return True

    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        """Project a and add to b."""
        projected_a = self.projection(a)
        return projected_a + b

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.projection.parameters()

    def to(self, device: torch.device) -> "LearnedLinearCombination":
        self.projection = self.projection.to(device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        return {"projection": self.projection.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "projection" in state_dict:
            self.projection.load_state_dict(state_dict["projection"])


class ConcatProjectCombination(CombinationFunction):
    """
    f(a, b) = MLP([W_a @ a; W_b @ b])

    Concatenates projected activations and passes through MLP.
    Preserves more information than purely additive combinations.
    """

    def __init__(
        self,
        d_a: int,
        d_b: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize concat+project combination.

        Args:
            d_a: Dimension of Model A's activations
            d_b: Dimension of Model B's activations
            hidden_dim: Hidden dimension (defaults to d_b)
            dropout: Dropout rate in MLP
        """
        self.d_a = d_a
        self.d_b = d_b
        hidden_dim = hidden_dim or d_b

        self.proj_a = nn.Linear(d_a, hidden_dim, bias=False)
        self.proj_b = nn.Linear(d_b, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_b),
        )

        # Initialize
        for module in [self.proj_a, self.proj_b]:
            nn.init.normal_(module.weight, std=0.02)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def trainable(self) -> bool:
        return True

    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        proj_a = self.proj_a(a)
        proj_b = self.proj_b(b)
        concat = torch.cat([proj_a, proj_b], dim=-1)
        return self.mlp(concat)

    def parameters(self) -> Iterator[nn.Parameter]:
        yield from self.proj_a.parameters()
        yield from self.proj_b.parameters()
        yield from self.mlp.parameters()

    def to(self, device: torch.device) -> "ConcatProjectCombination":
        self.proj_a = self.proj_a.to(device)
        self.proj_b = self.proj_b.to(device)
        self.mlp = self.mlp.to(device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        return {
            "proj_a": self.proj_a.state_dict(),
            "proj_b": self.proj_b.state_dict(),
            "mlp": self.mlp.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.proj_a.load_state_dict(state_dict["proj_a"])
        self.proj_b.load_state_dict(state_dict["proj_b"])
        self.mlp.load_state_dict(state_dict["mlp"])


class GatedCombination(CombinationFunction):
    """
    f(a, b) = gate * project(a) + (1 - gate) * b

    where gate = sigmoid(W_g @ [a; b])

    Learnable gating mechanism that decides how much of A's
    information to incorporate based on both activations.
    """

    def __init__(self, d_a: int, d_b: int):
        """
        Initialize gated combination.

        Args:
            d_a: Dimension of Model A's activations
            d_b: Dimension of Model B's activations
        """
        self.d_a = d_a
        self.d_b = d_b

        self.projection = nn.Linear(d_a, d_b, bias=False)
        self.gate = nn.Linear(d_a + d_b, d_b, bias=True)

        # Initialize projection small, gate bias negative (favor B initially)
        nn.init.normal_(self.projection.weight, std=0.02)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.constant_(self.gate.bias, -2.0)  # sigmoid(-2) ≈ 0.12

    @property
    def trainable(self) -> bool:
        return True

    def combine(self, a: Tensor, b: Tensor) -> Tensor:
        projected_a = self.projection(a)
        gate_input = torch.cat([a, b], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))
        return gate_values * projected_a + (1 - gate_values) * b

    def parameters(self) -> Iterator[nn.Parameter]:
        yield from self.projection.parameters()
        yield from self.gate.parameters()

    def to(self, device: torch.device) -> "GatedCombination":
        self.projection = self.projection.to(device)
        self.gate = self.gate.to(device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        return {
            "projection": self.projection.state_dict(),
            "gate": self.gate.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.projection.load_state_dict(state_dict["projection"])
        self.gate.load_state_dict(state_dict["gate"])


def get_combination_function(  # pylint: disable=too-many-return-statements
    name: str,
    d_a: int,
    d_b: int,
    **kwargs,
) -> CombinationFunction:
    """
    Factory function to create combination functions by name.

    Args:
        name: Name of the combination function
        d_a: Dimension of Model A's activations
        d_b: Dimension of Model B's activations
        **kwargs: Additional arguments for specific functions

    Returns:
        CombinationFunction instance
    """
    name = name.lower().replace("-", "_")

    if name == "sum":
        if d_a != d_b:
            raise ValueError(
                f"SumCombination requires d_a == d_b, got {d_a} vs {d_b}"
            )
        return SumCombination()

    if name == "mean":
        if d_a != d_b:
            raise ValueError(
                f"MeanCombination requires d_a == d_b, got {d_a} vs {d_b}"
            )
        return MeanCombination()

    if name == "replace":
        return ReplaceCombination(d_a, d_b)

    if name == "weighted_sum":
        if d_a != d_b:
            raise ValueError(
                f"WeightedSumCombination requires d_a == d_b, "
                f"got {d_a} vs {d_b}"
            )
        return WeightedSumCombination(
            init_alpha=kwargs.get("init_alpha", 0.0)
        )

    if name in ["learned_linear", "linear"]:
        return LearnedLinearCombination(
            d_a, d_b, bias=kwargs.get("bias", False)
        )

    if name in ["concat_project", "concat"]:
        return ConcatProjectCombination(
            d_a, d_b,
            hidden_dim=kwargs.get("hidden_dim"),
            dropout=kwargs.get("dropout", 0.1),
        )

    if name == "gated":
        return GatedCombination(d_a, d_b)

    if name in ["trained_projection", "trained"]:
        projection_path = kwargs.get("projection_path")
        if not projection_path:
            raise ValueError(
                "trained_projection requires 'projection_path' kwarg"
            )
        return TrainedProjectionCombination(
            projection_path,
            normalize_source=kwargs.get("normalize_source", False),
        )

    if name in ["random_noise"]:
        std = kwargs.get("std", 0.01)
        return RandomNoiseCombination(std=std)

    if name in ["random_add"]:
        std = kwargs.get("std", 0.01)
        return RandomAddCombination(std=std)

    if name in ["zero_vector", "zero"]:
        return ZeroVectorCombination()

    if name in ["average_act"]:
        return AverageActivationCombination()

    if name in ["random_unit"]:
        return RandomUnitCombination()

    if name in ["shuffled_act"]:
        return ShuffledActivationCombination()

    if name in ["bos_token"]:
        return BosTokenCombination()

    if name in ["prev_layer"]:
        return PrevLayerActivationCombination()

    if name in ["projected_average"]:
        projection_path = kwargs.get("projection_path")
        if not projection_path:
            raise ValueError(
                "projected_average requires 'projection_path' kwarg"
            )
        alpha = kwargs.get("alpha", 0.5)
        return ProjectedAverageCombination(projection_path, alpha=alpha)

    if name in ["projected_sum"]:
        projection_path = kwargs.get("projection_path")
        if not projection_path:
            raise ValueError(
                "projected_sum requires 'projection_path' kwarg"
            )
        return ProjectedSumCombination(
            projection_path,
            normalize_source=kwargs.get("normalize_source", False),
        )

    available = [
        "sum", "mean", "replace", "weighted_sum",
        "learned_linear", "concat_project", "gated",
        "trained_projection", "projected_average", "projected_sum",
        "random_noise", "random_add", "zero_vector",
    ]
    raise ValueError(
        f"Unknown combination function: {name}. Available: {available}"
    )


if __name__ == "__main__":
    # Test combination functions
    print("Testing combination functions...")

    batch_size = 2
    dim_a = 896
    dim_b = 1536

    act_a = torch.randn(batch_size, dim_a)
    act_b = torch.randn(batch_size, dim_b)

    # Same dimension tests
    act_a_same = torch.randn(batch_size, dim_b)

    print("\nSame dimension (d=1536):")
    for fn_name in ["sum", "mean", "weighted_sum"]:
        fn = get_combination_function(fn_name, dim_b, dim_b)
        out = fn.combine(act_a_same, act_b)
        print(f"  {fn_name}: output shape = {out.shape}")

    print("\nDifferent dimensions (896 -> 1536):")
    for fn_name in ["replace", "learned_linear", "concat_project", "gated"]:
        fn = get_combination_function(fn_name, dim_a, dim_b)
        out = fn.combine(act_a, act_b)
        print(f"  {fn_name}: output shape = {out.shape}, trainable = {fn.trainable}")

    print("\nTest passed!")
