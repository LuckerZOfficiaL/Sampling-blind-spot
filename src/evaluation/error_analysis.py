"""Error analysis tools for understanding model failures."""

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Tuple
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


@dataclass
class PredictionRecord:
    """
    Record of a single prediction for error analysis.

    Stores inputs, outputs, configuration, and diagnostic information
    to enable post-hoc analysis of model behavior.
    """

    # Task identification
    task_id: str
    benchmark: str

    # Inputs
    input_a: str
    input_b: str
    expected: str

    # Outputs
    predicted: str
    correct: bool

    # Configuration
    layer_a: int
    layer_b: int
    combination_fn: str
    grafting_mode: str
    extraction_point: str = "POST_MLP"

    # Diagnostics (optional)
    activation_a_norm: Optional[float] = None
    activation_b_norm: Optional[float] = None
    combined_norm: Optional[float] = None
    cosine_similarity_ab: Optional[float] = None
    output_entropy: Optional[float] = None

    # Metadata
    input_a_tokens: Optional[int] = None
    input_b_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    generation_time_ms: Optional[float] = None

    # Additional fields
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ErrorAnalyzer:
    """
    Collect and analyze prediction records to understand model failures.

    Provides functionality for:
    - Tracking predictions across experiments
    - Computing accuracy by various dimensions
    - Identifying error patterns
    - Correlation analysis with diagnostics
    """

    def __init__(self):
        self.records: List[PredictionRecord] = []

    def add_record(self, record: PredictionRecord) -> None:
        """Add a single prediction record."""
        self.records.append(record)

    def add(
        self,
        task_id: str,
        benchmark: str,
        input_a: str,
        input_b: str,
        expected: str,
        predicted: str,
        correct: bool,
        layer_a: int,
        layer_b: int,
        combination_fn: str,
        grafting_mode: str,
        **kwargs,
    ) -> None:
        """Add a record with individual fields."""
        record = PredictionRecord(
            task_id=task_id,
            benchmark=benchmark,
            input_a=input_a,
            input_b=input_b,
            expected=expected,
            predicted=predicted,
            correct=correct,
            layer_a=layer_a,
            layer_b=layer_b,
            combination_fn=combination_fn,
            grafting_mode=grafting_mode,
            **kwargs,
        )
        self.records.append(record)

    def save(self, path: str) -> None:
        """
        Save records to JSON file.

        Args:
            path: Output file path
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding="utf-8") as out_file:
            json.dump([r.to_dict() for r in self.records], out_file, indent=2)

    def load(self, path: str) -> None:
        """
        Load records from JSON file.

        Args:
            path: Input file path
        """
        with open(path, 'r', encoding="utf-8") as in_file:
            data = json.load(in_file)

        self.records = []
        for item in data:
            # Handle extra field separately
            extra = item.pop('extra', {})
            record = PredictionRecord(**item, extra=extra)
            self.records.append(record)

    def accuracy(self) -> float:
        """Compute overall accuracy."""
        if not self.records:
            return 0.0
        return sum(r.correct for r in self.records) / len(self.records)

    def accuracy_by_benchmark(self) -> Dict[str, float]:
        """Compute accuracy broken down by benchmark."""
        by_benchmark = defaultdict(lambda: {"correct": 0, "total": 0})

        for r in self.records:
            by_benchmark[r.benchmark]["total"] += 1
            if r.correct:
                by_benchmark[r.benchmark]["correct"] += 1

        return {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
            for k, v in by_benchmark.items()
        }

    def accuracy_by_layer_pair(self) -> Dict[Tuple[int, int], float]:
        """Compute accuracy by (layer_a, layer_b) pair."""
        by_layers = defaultdict(lambda: {"correct": 0, "total": 0})

        for r in self.records:
            key = (r.layer_a, r.layer_b)
            by_layers[key]["total"] += 1
            if r.correct:
                by_layers[key]["correct"] += 1

        return {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
            for k, v in by_layers.items()
        }

    def accuracy_by_combination_fn(self) -> Dict[str, float]:
        """Compute accuracy by combination function."""
        by_fn = defaultdict(lambda: {"correct": 0, "total": 0})

        for r in self.records:
            by_fn[r.combination_fn]["total"] += 1
            if r.correct:
                by_fn[r.combination_fn]["correct"] += 1

        return {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
            for k, v in by_fn.items()
        }

    def accuracy_by_input_length(
        self,
        field: str = "input_a_tokens",  # pylint: disable=redefined-outer-name
        n_bins: int = 5,
    ) -> Dict[str, float]:
        """
        Compute accuracy binned by input length.

        Args:
            field: Which length field to use
            n_bins: Number of bins

        Returns:
            Dictionary mapping bin labels to accuracy
        """
        # Get lengths
        lengths = []
        for r in self.records:
            length = getattr(r, field, None)
            if length is not None:
                lengths.append((length, r.correct))

        if not lengths:
            return {}

        # Create bins
        all_lengths = [l for l, _ in lengths]
        bin_edges = np.linspace(min(all_lengths), max(all_lengths) + 1, n_bins + 1)

        results = {}
        for bin_idx in range(n_bins):
            low, high = bin_edges[bin_idx], bin_edges[bin_idx + 1]
            in_bin = [(l, c) for l, c in lengths if low <= l < high]

            if in_bin:
                acc = sum(c for _, c in in_bin) / len(in_bin)
                label = f"{int(low)}-{int(high)}"
                results[label] = acc

        return results

    def error_examples(self, n: int = 10) -> List[PredictionRecord]:
        """
        Get example errors.

        Args:
            n: Maximum number of examples

        Returns:
            List of incorrect prediction records
        """
        errors = [r for r in self.records if not r.correct]
        return errors[:n]

    def correct_examples(self, n: int = 10) -> List[PredictionRecord]:
        """
        Get example correct predictions.

        Args:
            n: Maximum number of examples

        Returns:
            List of correct prediction records
        """
        correct = [r for r in self.records if r.correct]
        return correct[:n]

    def correlation_with_diagnostics(self) -> Dict[str, float]:
        """
        Compute correlation between diagnostic metrics and correctness.

        Returns:
            Dictionary mapping diagnostic names to correlation coefficients
        """
        diagnostic_fields = [
            "activation_a_norm",
            "activation_b_norm",
            "combined_norm",
            "cosine_similarity_ab",
            "output_entropy",
        ]

        correct = np.array([float(r.correct) for r in self.records])
        results = {}

        for diag_field in diagnostic_fields:
            values = [getattr(r, diag_field) for r in self.records]

            # Only compute if we have values
            if all(v is not None for v in values):
                values = np.array(values)

                # Check for constant arrays
                if values.std() > 1e-10 and correct.std() > 1e-10:
                    corr = np.corrcoef(correct, values)[0, 1]
                    if not np.isnan(corr):
                        results[diag_field] = corr

        return results

    def summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of results.

        Returns:
            Dictionary with summary statistics
        """
        return {
            "n_samples": len(self.records),
            "n_correct": sum(r.correct for r in self.records),
            "n_errors": sum(not r.correct for r in self.records),
            "accuracy": self.accuracy(),
            "by_benchmark": self.accuracy_by_benchmark(),
            "by_combination_fn": self.accuracy_by_combination_fn(),
            "diagnostic_correlations": self.correlation_with_diagnostics(),
        }

    def filter(
        self,
        benchmark: Optional[str] = None,
        layer_a: Optional[int] = None,
        layer_b: Optional[int] = None,
        combination_fn: Optional[str] = None,
        correct_only: bool = False,
        errors_only: bool = False,
    ) -> "ErrorAnalyzer":
        """
        Filter records and return new analyzer.

        Args:
            benchmark: Filter by benchmark name
            layer_a: Filter by layer A
            layer_b: Filter by layer B
            combination_fn: Filter by combination function
            correct_only: Only include correct predictions
            errors_only: Only include errors

        Returns:
            New ErrorAnalyzer with filtered records
        """
        filtered = ErrorAnalyzer()

        for r in self.records:
            if benchmark is not None and r.benchmark != benchmark:
                continue
            if layer_a is not None and r.layer_a != layer_a:
                continue
            if layer_b is not None and r.layer_b != layer_b:
                continue
            if combination_fn is not None and r.combination_fn != combination_fn:
                continue
            if correct_only and not r.correct:
                continue
            if errors_only and r.correct:
                continue

            filtered.records.append(r)

        return filtered

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd  # pylint: disable=import-outside-toplevel
            return pd.DataFrame([r.to_dict() for r in self.records])
        except ImportError as exc:
            raise ImportError("pandas required for to_dataframe()") from exc

    def print_summary(self) -> None:
        """Print a formatted summary."""
        summary = self.summary()

        print("=" * 60)
        print("Error Analysis Summary")
        print("=" * 60)
        print(f"Total samples: {summary['n_samples']}")
        accuracy_pct = summary['accuracy'] * 100
        print(f"Correct: {summary['n_correct']} ({accuracy_pct:.1f}%)")
        print(f"Errors: {summary['n_errors']}")

        print("\nBy Benchmark:")
        for bench, acc in summary['by_benchmark'].items():
            count = sum(1 for r in self.records if r.benchmark == bench)
            print(f"  {bench}: {acc * 100:.1f}% (n={count})")

        print("\nBy Combination Function:")
        for fn, acc in summary['by_combination_fn'].items():
            count = sum(1 for r in self.records if r.combination_fn == fn)
            print(f"  {fn}: {acc * 100:.1f}% (n={count})")

        if summary['diagnostic_correlations']:
            print("\nDiagnostic Correlations with Correctness:")
            for diag, corr in summary['diagnostic_correlations'].items():
                print(f"  {diag}: {corr:.3f}")

    def reset(self) -> None:
        """Reset the analyzer."""
        self.records = []


if __name__ == "__main__":
    # Test error analysis
    print("Testing error analysis...")

    analyzer = ErrorAnalyzer()

    # Add some test records
    for i in range(20):
        analyzer.add(
            task_id=f"test_{i}",
            benchmark="countries" if i < 10 else "gsm8k",
            input_a=f"Input A {i}",
            input_b=f"Input B {i}",
            expected=f"Expected {i}",
            predicted=f"Expected {i}" if i % 3 != 0 else f"Wrong {i}",
            correct=(i % 3 != 0),
            layer_a=12,
            layer_b=12,
            combination_fn="sum" if i < 15 else "learned_linear",
            grafting_mode="SINGLE_SHOT",
            activation_a_norm=float(i),
            cosine_similarity_ab=0.5 + 0.01 * i,
        )

    # Test methods
    print(f"  Accuracy: {analyzer.accuracy()*100:.1f}%")
    print(f"  By benchmark: {analyzer.accuracy_by_benchmark()}")
    print(f"  Errors: {len(analyzer.error_examples(5))}")

    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        analyzer.save(f.name)
        analyzer2 = ErrorAnalyzer()
        analyzer2.load(f.name)
        assert len(analyzer2.records) == len(analyzer.records)

    print("  Save/load: OK")

    # Print summary
    print()
    analyzer.print_summary()

    print("\nAll tests passed!")
