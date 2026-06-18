"""Evaluation metrics for benchmarks."""

import re
from typing import List, Optional, Tuple
from collections import Counter
import numpy as np


def normalize_answer(s: str) -> str:
    """
    Normalize answer string for comparison.

    Performs:
    - Lowercase
    - Strip whitespace
    - Remove punctuation
    - Remove articles (a, an, the)
    - Collapse whitespace
    """
    if not s:
        return ""

    s = s.lower().strip()

    # Remove punctuation
    s = re.sub(r'[^\w\s]', ' ', s)

    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)

    # Collapse whitespace
    s = ' '.join(s.split())

    return s


def exact_match(prediction: str, ground_truth: str) -> bool:
    """
    Check for exact match after normalization.

    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer

    Returns:
        True if predictions match after normalization
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def exact_match_any(prediction: str, ground_truths: List[str]) -> bool:
    """
    Check if prediction matches any of the ground truths.

    Args:
        prediction: Model prediction
        ground_truths: List of acceptable answers

    Returns:
        True if prediction matches any ground truth
    """
    norm_pred = normalize_answer(prediction)
    return any(norm_pred == normalize_answer(gt) for gt in ground_truths)


def contains_answer(prediction: str, ground_truth: str) -> bool:
    """
    Check if the ground truth appears within the prediction.

    More lenient than exact match - useful when model generates
    additional context around the answer.

    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer

    Returns:
        True if normalized ground truth is substring of normalized prediction
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)

    if not norm_gt:
        return True

    return norm_gt in norm_pred


def extract_number(s: str) -> Optional[float]:
    """
    Extract the last number from a string.

    Handles:
    - Integers and decimals
    - Negative numbers
    - Numbers with commas (e.g., "1,234")
    - Numbers at end of string (common in GSM8k)

    Args:
        s: String potentially containing numbers

    Returns:
        Extracted number or None if no number found
    """
    if not s:
        return None

    # Remove commas from numbers
    s = s.replace(',', '')

    # Find all numbers (including negative and decimals)
    numbers = re.findall(r'-?\d+\.?\d*', s)

    if numbers:
        try:
            return float(numbers[-1])  # Return last number
        except ValueError:
            return None

    return None


def numerical_match(
    prediction: str,
    ground_truth: str,
    tolerance: float = 1e-5,
) -> bool:
    """
    Check if extracted numbers match.

    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        tolerance: Numerical tolerance for comparison

    Returns:
        True if extracted numbers match within tolerance
    """
    pred_num = extract_number(prediction)
    gt_num = extract_number(ground_truth)

    if pred_num is None or gt_num is None:
        return False

    return abs(pred_num - gt_num) < tolerance


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score.

    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer

    Returns:
        F1 score between 0 and 1
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def batch_exact_match(
    predictions: List[str],
    ground_truths: List[str],
) -> float:
    """
    Compute exact match accuracy over a batch.

    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers

    Returns:
        Accuracy (fraction correct)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truths)} ground truths"
        )

    if not predictions:
        return 0.0

    correct = sum(
        exact_match(p, g) for p, g in zip(predictions, ground_truths)
    )
    return correct / len(predictions)


def batch_numerical_match(
    predictions: List[str],
    ground_truths: List[str],
    tolerance: float = 1e-5,
) -> float:
    """
    Compute numerical match accuracy over a batch.

    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers
        tolerance: Numerical tolerance

    Returns:
        Accuracy (fraction correct)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truths)} ground truths"
        )

    if not predictions:
        return 0.0

    correct = sum(
        numerical_match(p, g, tolerance)
        for p, g in zip(predictions, ground_truths)
    )
    return correct / len(predictions)


def batch_f1_score(
    predictions: List[str],
    ground_truths: List[str],
) -> float:
    """
    Compute average F1 score over a batch.

    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers

    Returns:
        Average F1 score
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truths)} ground truths"
        )

    if not predictions:
        return 0.0

    scores = [
        f1_score(p, g) for p, g in zip(predictions, ground_truths)
    ]
    return sum(scores) / len(scores)


def bootstrap_confidence_interval(
    predictions: List[str],
    ground_truths: List[str],
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers
        metric_fn: Function taking (predictions, ground_truths) -> float
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    np.random.seed(seed)

    n = len(predictions)
    scores = []

    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        sample_preds = [predictions[i] for i in indices]
        sample_gts = [ground_truths[i] for i in indices]
        score = metric_fn(sample_preds, sample_gts)
        scores.append(score)

    scores = np.array(scores)
    mean = scores.mean()

    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))

    return mean, lower, upper


def format_accuracy_with_ci(
    predictions: List[str],
    ground_truths: List[str],
    metric_fn=batch_exact_match,
    n_bootstrap: int = 1000,
) -> str:
    """
    Format accuracy with 95% confidence interval.

    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers
        metric_fn: Metric function
        n_bootstrap: Number of bootstrap samples

    Returns:
        Formatted string like "85.2% [82.1%, 88.0%]"
    """
    mean, lower, upper = bootstrap_confidence_interval(
        predictions, ground_truths, metric_fn, n_bootstrap
    )

    return f"{mean*100:.1f}% [{lower*100:.1f}%, {upper*100:.1f}%]"


class MetricTracker:
    """Track metrics across multiple evaluations."""

    def __init__(self):
        self.predictions: List[str] = []
        self.ground_truths: List[str] = []
        self.task_ids: List[str] = []

    def add(
        self,
        prediction: str,
        ground_truth: str,
        task_id: Optional[str] = None,
    ) -> None:
        """Add a single prediction."""
        self.predictions.append(prediction)
        self.ground_truths.append(ground_truth)
        self.task_ids.append(task_id or str(len(self.predictions)))

    def add_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
        task_ids: Optional[List[str]] = None,
    ) -> None:
        """Add a batch of predictions."""
        if task_ids is None:
            start = len(self.predictions)
            task_ids = [
                str(i) for i in range(start, start + len(predictions))
            ]

        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)
        self.task_ids.extend(task_ids)

    def exact_match_accuracy(self) -> float:
        """Get exact match accuracy."""
        return batch_exact_match(self.predictions, self.ground_truths)

    def numerical_accuracy(self) -> float:
        """Get numerical match accuracy."""
        return batch_numerical_match(self.predictions, self.ground_truths)

    def f1(self) -> float:
        """Get average F1 score."""
        return batch_f1_score(self.predictions, self.ground_truths)

    def accuracy_with_ci(self, metric: str = "exact_match") -> str:
        """Get formatted accuracy with confidence interval."""
        if metric == "exact_match":
            fn = batch_exact_match
        elif metric == "numerical":
            fn = batch_numerical_match
        elif metric == "f1":
            fn = batch_f1_score
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return format_accuracy_with_ci(self.predictions, self.ground_truths, fn)

    def get_errors(self) -> List[Tuple[str, str, str, str]]:
        """Get list of (task_id, prediction, ground_truth, normalized_pred) for errors."""
        errors = []
        for task_id, pred, gt in zip(self.task_ids, self.predictions, self.ground_truths):
            if not exact_match(pred, gt):
                errors.append((task_id, pred, gt, normalize_answer(pred)))
        return errors

    def summary(self) -> dict:
        """Get summary statistics."""
        return {
            "n_samples": len(self.predictions),
            "exact_match": self.exact_match_accuracy(),
            "numerical": self.numerical_accuracy(),
            "f1": self.f1(),
            "n_errors": len(self.get_errors()),
        }

    def reset(self) -> None:
        """Reset the tracker."""
        self.predictions = []
        self.ground_truths = []
        self.task_ids = []


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")

    # Exact match
    assert exact_match("France", "france") is True
    assert exact_match("The France", "France") is True
    assert exact_match("Paris", "France") is False
    print("  exact_match: OK")

    # Numerical match
    assert numerical_match("The answer is 42", "42") is True
    assert numerical_match("1,234", "1234") is True
    assert numerical_match("3.14159", "3.14159") is True
    assert numerical_match("42", "43") is False
    print("  numerical_match: OK")

    # F1 score
    assert f1_score("the quick brown fox", "quick brown fox") > 0.8
    assert f1_score("completely different", "nothing alike") < 0.1
    print("  f1_score: OK")

    # Batch metrics
    preds = ["France", "Germany", "Italy"]
    gts = ["France", "Germany", "Spain"]
    assert batch_exact_match(preds, gts) == 2/3
    print("  batch_exact_match: OK")

    print("\nAll tests passed!")
