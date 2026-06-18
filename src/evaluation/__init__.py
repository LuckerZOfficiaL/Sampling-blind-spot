"""Evaluation metrics and analysis tools."""

from src.evaluation.metrics import (
    normalize_answer,
    exact_match,
    exact_match_any,
    contains_answer,
    extract_number,
    numerical_match,
    f1_score,
    batch_exact_match,
    batch_numerical_match,
    batch_f1_score,
    bootstrap_confidence_interval,
    format_accuracy_with_ci,
    MetricTracker,
)
from src.evaluation.diagnostics import (
    cosine_similarity_stats,
    l2_distance_stats,
    centered_kernel_alignment,
    activation_statistics,
    compare_output_distributions,
)
from src.evaluation.error_analysis import (
    PredictionRecord,
    ErrorAnalyzer,
)

__all__ = [
    "normalize_answer",
    "exact_match",
    "exact_match_any",
    "contains_answer",
    "extract_number",
    "numerical_match",
    "f1_score",
    "batch_exact_match",
    "batch_numerical_match",
    "batch_f1_score",
    "bootstrap_confidence_interval",
    "format_accuracy_with_ci",
    "MetricTracker",
    "cosine_similarity_stats",
    "l2_distance_stats",
    "centered_kernel_alignment",
    "activation_statistics",
    "compare_output_distributions",
    "PredictionRecord",
    "ErrorAnalyzer",
]
