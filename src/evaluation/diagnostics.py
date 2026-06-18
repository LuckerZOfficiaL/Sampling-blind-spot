"""Diagnostic tools for analyzing activations and grafting behavior."""

from typing import Dict, List, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np


def cosine_similarity_stats(a: Tensor, b: Tensor) -> Dict[str, float]:
    """
    Compute cosine similarity statistics between activation sets.

    Args:
        a: Activations from source, shape [N, d_model]
        b: Activations from target, shape [N, d_model]

    Returns:
        Dictionary with mean, std, min, max of cosine similarities
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)

    sims = F.cosine_similarity(a, b, dim=-1)

    return {
        "mean": sims.mean().item(),
        "std": sims.std().item() if len(sims) > 1 else 0.0,
        "min": sims.min().item(),
        "max": sims.max().item(),
    }


def l2_distance_stats(a: Tensor, b: Tensor) -> Dict[str, float]:
    """
    Compute L2 distance statistics between activations.

    Args:
        a: Activations from source, shape [N, d_model]
        b: Activations from target, shape [N, d_model]

    Returns:
        Dictionary with mean, std, min, max of L2 distances
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)

    dists = (a - b).norm(dim=-1)

    return {
        "mean": dists.mean().item(),
        "std": dists.std().item() if len(dists) > 1 else 0.0,
        "min": dists.min().item(),
        "max": dists.max().item(),
    }


def centered_kernel_alignment(a: Tensor, b: Tensor) -> float:
    """
    Compute CKA (Centered Kernel Alignment) between two activation matrices.

    CKA measures the similarity between representations and is invariant
    to orthogonal transformations and isotropic scaling. It's commonly
    used to compare neural network representations.

    Args:
        a: Activations, shape [N, d_a]
        b: Activations, shape [N, d_b]

    Returns:
        CKA value between 0 and 1
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)

    # Center the data
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)

    # Compute linear kernel (Gram matrices)
    K_a = a @ a.T  # pylint: disable=invalid-name
    K_b = b @ b.T  # pylint: disable=invalid-name

    # HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_ab = (K_a * K_b).sum()
    hsic_aa = (K_a * K_a).sum()
    hsic_bb = (K_b * K_b).sum()

    # CKA
    denominator = torch.sqrt(hsic_aa * hsic_bb)
    if denominator < 1e-10:
        return 0.0

    cka = hsic_ab / denominator
    return cka.item()


def activation_statistics(activations: Tensor) -> Dict[str, float]:
    """
    Compute summary statistics of an activation tensor.

    Args:
        activations: Activation tensor, shape [N, d_model] or [d_model]

    Returns:
        Dictionary with various statistics
    """
    if activations.dim() == 1:
        activations = activations.unsqueeze(0)

    norms = activations.norm(dim=-1)

    return {
        "mean": activations.mean().item(),
        "std": activations.std().item(),
        "min": activations.min().item(),
        "max": activations.max().item(),
        "norm_mean": norms.mean().item(),
        "norm_std": norms.std().item() if len(norms) > 1 else 0.0,
        "norm_min": norms.min().item(),
        "norm_max": norms.max().item(),
        "sparsity": (activations.abs() < 0.01).float().mean().item(),
        "d_model": activations.shape[-1],
    }


def compare_output_distributions(
    logits_baseline: Tensor,
    logits_grafted: Tensor,
    top_k: int = 10,
) -> Dict[str, float]:
    """
    Compare output distributions with and without grafting.

    Args:
        logits_baseline: Logits without grafting, shape [vocab_size] or [batch, vocab_size]
        logits_grafted: Logits with grafting, same shape
        top_k: Number of top tokens to consider for overlap

    Returns:
        Dictionary with distribution comparison metrics
    """
    if logits_baseline.dim() == 1:
        logits_baseline = logits_baseline.unsqueeze(0)
    if logits_grafted.dim() == 1:
        logits_grafted = logits_grafted.unsqueeze(0)

    # Convert to probabilities
    probs_base = F.softmax(logits_baseline, dim=-1)
    probs_graft = F.softmax(logits_grafted, dim=-1)

    # KL divergence: KL(graft || base)
    # Note: KL can be unstable, add small epsilon
    eps = 1e-10
    kl_div = (
        probs_graft * (torch.log(probs_graft + eps) - torch.log(probs_base + eps))
    ).sum(dim=-1).mean().item()

    # JS divergence (symmetric)
    m = 0.5 * (probs_base + probs_graft)
    js_base = (
        probs_base * (torch.log(probs_base + eps) - torch.log(m + eps))
    ).sum(dim=-1)
    js_graft = (
        probs_graft * (torch.log(probs_graft + eps) - torch.log(m + eps))
    ).sum(dim=-1)
    js_div = (0.5 * (js_base + js_graft)).mean().item()

    # Top-k overlap
    top_k_base = torch.topk(probs_base, k=top_k, dim=-1).indices
    top_k_graft = torch.topk(probs_graft, k=top_k, dim=-1).indices

    overlap = 0.0
    for i in range(top_k_base.shape[0]):
        base_set = set(top_k_base[i].tolist())
        graft_set = set(top_k_graft[i].tolist())
        overlap += len(base_set & graft_set) / top_k
    overlap /= top_k_base.shape[0]

    # Entropy
    entropy_base = -(
        probs_base * torch.log(probs_base + eps)
    ).sum(dim=-1).mean().item()
    entropy_graft = -(
        probs_graft * torch.log(probs_graft + eps)
    ).sum(dim=-1).mean().item()

    # Top token agreement
    top_token_base = probs_base.argmax(dim=-1)
    top_token_graft = probs_graft.argmax(dim=-1)
    top_token_agreement = (
        (top_token_base == top_token_graft).float().mean().item()
    )

    return {
        "kl_divergence": kl_div,
        "js_divergence": js_div,
        "top_k_overlap": overlap,
        "top_token_agreement": top_token_agreement,
        "entropy_baseline": entropy_base,
        "entropy_grafted": entropy_graft,
        "entropy_change": entropy_graft - entropy_base,
    }


def layer_similarity_matrix(
    activations_a: Dict[int, Tensor],
    activations_b: Dict[int, Tensor],
    metric: str = "cosine",
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Compute similarity matrix between layers of two models.

    Args:
        activations_a: Dict mapping layer_idx to activation tensor for model A
        activations_b: Dict mapping layer_idx to activation tensor for model B
        metric: Similarity metric ("cosine", "cka", "l2")

    Returns:
        Tuple of (similarity_matrix, layers_a, layers_b)
    """
    layers_a = sorted(activations_a.keys())
    layers_b = sorted(activations_b.keys())

    matrix = np.zeros((len(layers_a), len(layers_b)))

    for i, la in enumerate(layers_a):
        for j, lb in enumerate(layers_b):
            act_a = activations_a[la]
            act_b = activations_b[lb]

            if metric == "cosine":
                if act_a.shape[-1] == act_b.shape[-1]:
                    sim = F.cosine_similarity(
                        act_a.flatten(), act_b.flatten(), dim=0
                    ).item()
                else:
                    sim = 0.0  # Cannot compute cosine for different dimensions
            elif metric == "cka":
                sim = centered_kernel_alignment(act_a, act_b)
            elif metric == "l2":
                if act_a.shape == act_b.shape:
                    sim = -l2_distance_stats(act_a, act_b)["mean"]  # Negative for similarity
                else:
                    sim = float('-inf')
            else:
                raise ValueError(f"Unknown metric: {metric}")

            matrix[i, j] = sim

    return matrix, layers_a, layers_b


def analyze_grafting_effect(
    activation_a: Tensor,
    activation_b: Tensor,
    combined: Tensor,
) -> Dict[str, float]:
    """
    Analyze the effect of grafting on activations.

    Args:
        activation_a: Activation from Model A
        activation_b: Activation from Model B (before grafting)
        combined: Combined activation (after grafting)

    Returns:
        Dictionary with analysis metrics
    """
    # Ensure tensors are flattened for comparison
    a = activation_a.flatten()
    b = activation_b.flatten()
    c = combined.flatten()

    # Norms
    norm_a = a.norm().item()
    norm_b = b.norm().item()
    norm_c = c.norm().item()

    # Similarities
    if a.shape == b.shape:
        cos_ab = F.cosine_similarity(a, b, dim=0).item()
    else:
        cos_ab = float('nan')

    min_ac = min(len(a), len(c))
    cos_ac = (
        F.cosine_similarity(a[:min_ac], c[:min_ac], dim=0).item()
        if len(a) > 0 else 0
    )
    min_bc = min(len(b), len(c))
    cos_bc = (
        F.cosine_similarity(b[:min_bc], c[:min_bc], dim=0).item()
        if len(b) > 0 else 0
    )

    # How much did combined change from B?
    if b.shape == c.shape:
        change_from_b = (c - b).norm().item()
        relative_change = change_from_b / (norm_b + 1e-10)
    else:
        change_from_b = float('nan')
        relative_change = float('nan')

    return {
        "norm_a": norm_a,
        "norm_b": norm_b,
        "norm_combined": norm_c,
        "cosine_ab": cos_ab,
        "cosine_a_combined": cos_ac,
        "cosine_b_combined": cos_bc,
        "l2_change_from_b": change_from_b,
        "relative_change_from_b": relative_change,
        "norm_ratio_c_b": norm_c / (norm_b + 1e-10),
    }


class DiagnosticsCollector:
    """Collect and aggregate diagnostics across multiple samples."""

    def __init__(self):
        self.samples: List[Dict[str, float]] = []

    def add(self, diagnostics: Dict[str, float]) -> None:
        """Add a sample's diagnostics."""
        self.samples.append(diagnostics)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics across all samples.

        Returns:
            Dictionary mapping metric names to {mean, std, min, max}
        """
        if not self.samples:
            return {}

        # Get all keys
        keys = set()
        for s in self.samples:
            keys.update(s.keys())

        result = {}
        for key in keys:
            values = [s.get(key) for s in self.samples if s.get(key) is not None]
            if values:
                values = np.array(values)
                result[key] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }

        return result

    def to_dataframe(self):
        """Convert to pandas DataFrame (if available)."""
        try:
            import pandas as pd  # pylint: disable=import-outside-toplevel
            return pd.DataFrame(self.samples)
        except ImportError as exc:
            raise ImportError("pandas required for to_dataframe()") from exc

    def reset(self) -> None:
        """Reset the collector."""
        self.samples = []


if __name__ == "__main__":
    # Test diagnostics
    print("Testing diagnostics...")

    # Test cosine similarity
    test_a = torch.randn(10, 768)
    test_b = torch.randn(10, 768)

    stats = cosine_similarity_stats(test_a, test_b)
    print(f"  Cosine similarity stats: mean={stats['mean']:.4f}")

    # Test CKA
    cka_val = centered_kernel_alignment(test_a, test_b)
    print(f"  CKA: {cka_val:.4f}")

    # Test activation statistics
    act_stats = activation_statistics(test_a)
    print(f"  Activation stats: norm_mean={act_stats['norm_mean']:.4f}")

    # Test distribution comparison
    logits1 = torch.randn(1, 32000)
    logits2 = logits1 + torch.randn(1, 32000) * 0.1
    dist_stats = compare_output_distributions(logits1, logits2)
    print(f"  Distribution comparison: KL={dist_stats['kl_divergence']:.4f}")

    print("\nAll tests passed!")
