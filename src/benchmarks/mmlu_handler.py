"""MMLU handler for single-turn AC experiments."""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.benchmarks.single_turn_handler import SingleTurnBenchmarkHandler
from src.utils.wandb_utils import WandbRun

try:
    from datasets import load_dataset, load_from_disk
    _HF_DATASETS_AVAILABLE = True
except ImportError:
    _HF_DATASETS_AVAILABLE = False

_DEFAULT_DATA_DIR = "${DATASETS_DIR}/mmlu"

_CHOICE_LABELS = ["A", "B", "C", "D"]


def _load_mmlu_local(data_dir: str, split: str = "test") -> List[Dict[str, Any]]:
    """Load MMLU from a local HuggingFace-saved dataset directory."""
    try:
        dataset = load_from_disk(data_dir)
        if hasattr(dataset, "__getitem__") and split in dataset:
            dataset = dataset[split]
    except Exception:
        # Fallback: try load_dataset with local path (parquet layout)
        dataset = load_dataset(data_dir, split=split)
    return dataset


def _load_mmlu_hf(data_dir: str, split: str = "test") -> List[Dict[str, Any]]:
    """Download MMLU 'all' config from HuggingFace and save locally."""
    if not _HF_DATASETS_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets library is required. "
            "Install with: pip install datasets"
        )
    dataset = load_dataset("cais/mmlu", "all", split=split)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(data_dir)
    return dataset


def _build_entries(dataset, subjects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Convert HuggingFace MMLU dataset rows to entry dicts."""
    entries = []
    subject_counters: Dict[str, int] = {}

    for item in dataset:
        subject = item["subject"]
        if subjects is not None and subject not in subjects:
            continue

        idx = subject_counters.get(subject, 0)
        subject_counters[subject] = idx + 1

        entries.append({
            "id": f"mmlu_{subject}_{idx:04d}",
            "question": item["question"],
            "choices": item["choices"],
            "expected": _CHOICE_LABELS[item["answer"]],
            "subject": subject,
        })

    return entries


class MMLUHandler(SingleTurnBenchmarkHandler):
    """SingleTurnBenchmarkHandler implementation for MMLU."""

    def load_data(
        self, n_samples: int, seed: int, **kwargs,
    ) -> List[Dict[str, Any]]:
        """Load MMLU examples as dicts.

        Supports kwargs:
            data_dir: local path to saved dataset
                (default: ${DATASETS_DIR}/mmlu).
            subjects: list of subject strings to restrict to (default: all 57).
            split: dataset split to load (default: "test").
            match_samples_from: path to predictions.json to reuse sample IDs.
        """
        if not _HF_DATASETS_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets library is required. "
                "Install with: pip install datasets"
            )

        data_dir = kwargs.get("data_dir", _DEFAULT_DATA_DIR)
        subjects = kwargs.get("subjects", None)
        split = kwargs.get("split", "test")
        match_from = kwargs.get("match_samples_from")

        # Load raw dataset
        local_path = Path(data_dir)
        if local_path.exists():
            dataset = _load_mmlu_local(data_dir, split=split)
        else:
            raise FileNotFoundError(
                f"Local MMLU dataset not found at '{data_dir}'. "
                "Download it once on a login node with:\n"
                "  from datasets import load_dataset\n"
                "  ds = load_dataset('cais/mmlu', 'all', split='test')\n"
                f"  ds.save_to_disk('{data_dir}')"
            )

        entries = _build_entries(dataset, subjects=subjects)

        # Reproduce a previous run's sample IDs
        if match_from:
            with open(match_from, "r", encoding="utf-8") as f:
                prev = json.load(f)
            sample_ids = {p["id"] for p in prev}
            return [e for e in entries if e["id"] in sample_ids]

        # Sample evenly across subjects
        if n_samples is not None and len(entries) > n_samples:
            random.seed(seed)
            by_subject: Dict[str, List] = {}
            for e in entries:
                by_subject.setdefault(e["subject"], []).append(e)

            subject_list = sorted(by_subject.keys())
            n_subjects = len(subject_list)
            per_subject = max(1, n_samples // n_subjects)
            sampled = []
            for subj in subject_list:
                pool = by_subject[subj]
                k = min(per_subject, len(pool))
                sampled.extend(random.sample(pool, k))

            # Top up to exactly n_samples if rounding left us short
            if len(sampled) < n_samples:
                remaining = [e for e in entries if e not in sampled]
                sampled.extend(random.sample(remaining, n_samples - len(sampled)))

            entries = sampled[:n_samples]

        return entries

    def create_prompt(
        self, entry: Dict[str, Any], tokenizer: Any,
    ) -> str:
        """Format question + choices using the model's chat template."""
        choices_text = "\n".join(
            f"{label}. {choice}"
            for label, choice in zip(_CHOICE_LABELS, entry["choices"])
        )
        content = (
            "The following is a multiple choice question. "
            "Think step by step, then wrap your final answer as <answer>X</answer> "
            "where X is A, B, C, or D.\n\n"
            f"Question: {entry['question']}\n"
            f"{choices_text}\n\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": content}]

        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        return content

    def extract_answer(self, response: str) -> Optional[str]:
        """Extract a choice letter (A/B/C/D) from model response."""
        # Prefer explicit tag: <answer>X</answer>
        match = re.search(r"<answer>\s*([ABCD])\s*</answer>", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Fallback: bare letter at the start (e.g. "B" or "B.")
        match = re.match(r"^\s*([ABCD])\b", response.strip())
        if match:
            return match.group(1)

        # "answer is X" / "answer: X"
        match = re.search(
            r"(?:answer|option)\s*(?:is|:)\s*([ABCD])\b",
            response, re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()

        # Last standalone letter anywhere
        all_matches = re.findall(r"\b([ABCD])\b", response)
        if all_matches:
            return all_matches[-1].upper()

        return None

    def check_correct(
        self, predicted: Optional[str], expected: str,
    ) -> bool:
        """Check whether the predicted letter matches the expected letter."""
        if predicted is None:
            return False
        return predicted.upper() == expected.upper()

    def write_results(
        self, predictions: List[Dict[str, Any]], output_dir: str,
        experiment_name: str,
    ) -> Path:
        """Write predictions JSON and results summary with per-subject breakdown."""
        out = Path(output_dir) / experiment_name
        out.mkdir(parents=True, exist_ok=True)

        # Predictions
        pred_path = out / "predictions.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)

        # Overall accuracy
        n_correct = sum(1 for p in predictions if p["correct"])
        accuracy = n_correct / len(predictions) * 100 if predictions else 0
        ci_lower, ci_upper = self._bootstrap_ci(
            [1 if p["correct"] else 0 for p in predictions]
        )

        # Per-subject accuracy
        by_subject: Dict[str, List] = {}
        for p in predictions:
            by_subject.setdefault(p["subject"], []).append(p)

        subject_accuracy = {
            subj: sum(1 for p in preds if p["correct"]) / len(preds) * 100
            for subj, preds in sorted(by_subject.items())
        }

        results = {
            "experiment": experiment_name,
            "n_samples": len(predictions),
            "accuracy": accuracy,
            "accuracy_ci": f"{accuracy:.1f}% [{ci_lower:.1f}%, {ci_upper:.1f}%]",
            "n_correct": n_correct,
            "subject_accuracy": subject_accuracy,
        }
        with open(out / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return pred_path

    def log_wandb_results(
        self, wb_run: WandbRun, predictions: List[Dict[str, Any]],
        experiment_name: str,
    ) -> None:
        """Log predictions table, overall accuracy, and per-subject accuracy to W&B."""
        n_correct = sum(1 for p in predictions if p["correct"])
        accuracy = n_correct / len(predictions) * 100 if predictions else 0
        ci_lower, ci_upper = self._bootstrap_ci(
            [1 if p["correct"] else 0 for p in predictions]
        )

        # Per-subject accuracy
        by_subject: Dict[str, List] = {}
        for p in predictions:
            by_subject.setdefault(p["subject"], []).append(p)

        subject_summary = {
            f"accuracy_{subj}": sum(1 for p in preds if p["correct"]) / len(preds) * 100
            for subj, preds in sorted(by_subject.items())
        }

        wb_run.summary_update({
            "accuracy": accuracy,
            "accuracy_ci_lower": ci_lower,
            "accuracy_ci_upper": ci_upper,
            "n_correct": n_correct,
            "n_samples": len(predictions),
            **subject_summary,
        })

        wb_run.log_table(
            key="predictions",
            columns=["id", "subject", "question", "choices", "expected",
                     "extracted_answer", "correct"],
            data=[
                [
                    p["id"], p["subject"], p["question"][:200],
                    " / ".join(p["choices"]), p["expected"],
                    p.get("extracted_answer"), p["correct"],
                ]
                for p in predictions
            ],
        )

    @staticmethod
    def _bootstrap_ci(
        correct_list: List[int],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> tuple:
        """Compute bootstrap CI for accuracy."""
        np.random.seed(seed)
        arr = np.array(correct_list)
        n = len(arr)
        accs = []
        for _ in range(n_bootstrap):
            sample = arr[np.random.randint(0, n, size=n)]
            accs.append(sample.mean() * 100)
        accs = np.array(accs)
        alpha = 1 - confidence
        return (
            float(np.percentile(accs, 100 * alpha / 2)),
            float(np.percentile(accs, 100 * (1 - alpha / 2))),
        )
