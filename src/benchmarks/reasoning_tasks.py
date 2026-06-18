"""Reasoning task benchmarks (GSM8k, etc.)."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import re
import random
import os
from pathlib import Path

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


@dataclass
class ReasoningExample:
    """Single reasoning task example."""

    id: str
    question: str
    answer: str
    chain_of_thought: Optional[str] = None
    numerical_answer: Optional[float] = None


# =============================================================================
# GSM8k Dataset
# =============================================================================

def extract_numerical_answer(answer_text: str) -> Optional[float]:
    """
    Extract numerical answer from GSM8k-style answer.

    GSM8k answers end with "#### <number>".

    Args:
        answer_text: Full answer text

    Returns:
        Extracted number or None
    """
    # Look for #### pattern
    match = re.search(r'####\s*(-?\d+\.?\d*)', answer_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Fallback: find last number
    numbers = re.findall(r'-?\d+\.?\d*', answer_text.replace(',', ''))
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass

    return None


def load_gsm8k_from_huggingface(
    split: str = "test",
    n_samples: Optional[int] = 100,
    seed: int = 42,
) -> List[ReasoningExample]:
    """
    Load GSM8K dataset from HuggingFace.

    This loads the official GSM8K dataset used in the paper (arXiv:2501.14082).
    The paper uses a randomly-sampled size-100 subset for evaluation.

    Args:
        split: Dataset split ("train" or "test"). Default "test".
        n_samples: Number of samples to return. None for all samples.
        seed: Random seed for sampling.

    Returns:
        List of ReasoningExample instances

    Raises:
        ImportError: If datasets library is not installed
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets library is required. "
            "Install with: pip install datasets"
        )

    random.seed(seed)

    # Load GSM8K from HuggingFace
    # Dataset: https://huggingface.co/datasets/openai/gsm8k

    # Check if local dataset directory exists (for offline mode)
    local_data_dir = os.getenv("HF_DATASETS_CACHE")
    if local_data_dir:
        gsm8k_local = Path(local_data_dir) / "gsm8k"
        if gsm8k_local.exists():
            print(f"Loading GSM8K from local directory: {gsm8k_local}")
            # Try load_from_disk first (save_to_disk format with train/test subdirs)
            try:
                from datasets import load_from_disk
                ds = load_from_disk(str(gsm8k_local))
                dataset = ds[split] if hasattr(ds, "keys") else ds
            except Exception:
                # Fall back to parquet files (HuggingFace cache format)
                dataset = load_dataset("parquet", data_files={
                    "train": str(gsm8k_local / "main" / "train-*.parquet"),
                    "test": str(gsm8k_local / "main" / "test-*.parquet"),
                }, split=split)
        else:
            dataset = load_dataset("openai/gsm8k", "main", split=split)
    else:
        dataset = load_dataset("openai/gsm8k", "main", split=split)

    examples = []
    for i, item in enumerate(dataset):
        question = item["question"]
        answer_text = item["answer"]

        numerical = extract_numerical_answer(answer_text)
        cot = (
            answer_text.split('####')[0].strip()
            if '####' in answer_text else answer_text
        )

        examples.append(ReasoningExample(
            id=f"gsm8k_{i:04d}",
            question=question,
            answer=str(numerical) if numerical is not None else answer_text,
            chain_of_thought=cot,
            numerical_answer=numerical,
        ))

    # Sample if needed
    if n_samples is not None and len(examples) > n_samples:
        examples = random.sample(examples, n_samples)

    return examples


def load_gsm8k_subset(
    n_samples: Optional[int] = 100,
    seed: int = 42,
    split: str = "test",  # train or test
) -> List[ReasoningExample]:
    """
    Load a subset of GSM8k dataset.


    Args:
        n_samples: Number of samples to return
        seed: Random seed
        split: Dataset split ("train" or "test"). Default "test".

    Returns:
        List of ReasoningExample instances
    """
    random.seed(seed)

    _examples = []

    try:
        return load_gsm8k_from_huggingface(
            split=split,
            n_samples=n_samples,
            seed=seed,
        )
    except Exception as e:
        print(f"Warning: Failed to load GSM8K from HuggingFace: {e}")
        raise


def save_gsm8k_subset(
    examples: List[ReasoningExample],
    path: str,
) -> None:
    """Save GSM8k subset to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    data = []
    for ex in examples:
        data.append({
            "id": ex.id,
            "question": ex.question,
            "answer": ex.answer,
            "chain_of_thought": ex.chain_of_thought,
            "numerical_answer": ex.numerical_answer,
        })

    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def create_gsm8k_prompts(example: ReasoningExample) -> Tuple[str, str]:
    """
    Create prompt_a and prompt_b for GSM8K coordination experiments.

    For reasoning tasks, both models see the same question.
    Model A reasons about the problem, Model B generates the answer.

    Args:
        example: ReasoningExample with question and answer

    Returns:
        Tuple of (prompt_a, prompt_b)
    """
    # Both models get the same question in coordination setup
    prompt = (
        f"Problem: {example.question}\n\n"
        "Solve step by step and give the final numerical answer:"
    )
    return prompt, prompt


def load_gsm8k_from_json(path: str) -> List[ReasoningExample]:
    """Load GSM8k examples from our JSON format."""
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    return [
        ReasoningExample(
            id=item["id"],
            question=item["question"],
            answer=item["answer"],
            chain_of_thought=item.get("chain_of_thought"),
            numerical_answer=item.get("numerical_answer"),
        )
        for item in data
    ]


# =============================================================================
# MMLU Dataset
# =============================================================================

_MMLU_CHOICE_LABELS = ["A", "B", "C", "D"]
_DEFAULT_MMLU_DATA_DIR = os.environ.get("DATASETS_DIR", "") and os.path.join(os.environ["DATASETS_DIR"], "mmlu") or None

# =============================================================================
# MMLU-Pro Dataset
# =============================================================================

_MMLU_PRO_CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
_DEFAULT_MMLU_PRO_DATA_DIR = os.environ.get("DATASETS_DIR", "") and os.path.join(os.environ["DATASETS_DIR"], "mmlu_pro") or None

# =============================================================================
# MATH Dataset (Hendrycks et al.)
# =============================================================================

_DEFAULT_MATH_DATA_DIR = os.environ.get("DATASETS_DIR", "") and os.path.join(os.environ["DATASETS_DIR"], "math") or None


@dataclass
class MMLUExample:
    """Single MMLU multiple-choice example."""

    id: str
    question: str
    choices: List[str]
    expected: str  # "A", "B", "C", or "D"
    subject: str


def load_mmlu_subset(
    n_samples: Optional[int] = 100,
    seed: int = 42,
    data_dir: str = _DEFAULT_MMLU_DATA_DIR,
    subjects: Optional[List[str]] = None,
    split: str = "test",
) -> List[MMLUExample]:
    """Load MMLU examples from local disk.

    Args:
        n_samples: Number of examples to return. None for all.
        seed: Random seed for even-across-subjects sampling.
        data_dir: Local path to MMLU dataset saved with save_to_disk().
        subjects: List of subjects to include (default: all 57).
        split: Dataset split to load (default: "test").

    Returns:
        List of MMLUExample instances.
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets library is required. "
            "Install with: pip install datasets"
        )

    from datasets import load_from_disk, load_dataset as _load_dataset  # pylint: disable=import-outside-toplevel

    local_path = Path(data_dir)
    if not local_path.exists():
        raise FileNotFoundError(
            f"Local MMLU dataset not found at '{data_dir}'. "
            "Download it once on a login node with:\n"
            "  from datasets import load_dataset\n"
            "  ds = load_dataset('cais/mmlu', 'all', split='test')\n"
            f"  ds.save_to_disk('{data_dir}')"
        )

    try:
        # Try loading the split subdirectory directly first (avoids DatasetDict metadata issues)
        split_path = local_path / split
        if split_path.is_dir():
            dataset = load_from_disk(str(split_path))
        else:
            dataset = load_from_disk(data_dir)
            if hasattr(dataset, "__getitem__") and split in dataset:
                dataset = dataset[split]
    except Exception:  # pylint: disable=broad-exception-caught
        dataset = _load_dataset(data_dir, split=split)

    subject_counters: Dict[str, int] = {}
    entries: List[MMLUExample] = []
    for item in dataset:
        subject = item["subject"]
        if subjects is not None and subject not in subjects:
            continue
        idx = subject_counters.get(subject, 0)
        subject_counters[subject] = idx + 1
        entries.append(MMLUExample(
            id=f"mmlu_{subject}_{idx:04d}",
            question=item["question"],
            choices=list(item["choices"]),
            expected=_MMLU_CHOICE_LABELS[item["answer"]],
            subject=subject,
        ))

    if n_samples is not None and len(entries) > n_samples:
        random.seed(seed)
        by_subject: Dict[str, List] = {}
        for e in entries:
            by_subject.setdefault(e.subject, []).append(e)

        subject_list = sorted(by_subject.keys())
        per_subject = max(1, n_samples // len(subject_list))
        sampled: List[MMLUExample] = []
        for subj in subject_list:
            pool = by_subject[subj]
            sampled.extend(random.sample(pool, min(per_subject, len(pool))))

        if len(sampled) < n_samples:
            remaining = [e for e in entries if e not in sampled]
            sampled.extend(random.sample(remaining, n_samples - len(sampled)))

        entries = sampled[:n_samples]

    return entries


def load_mmlu_pro_subset(
    n_samples: Optional[int] = 100,
    seed: int = 42,
    data_dir: str = _DEFAULT_MMLU_PRO_DATA_DIR,
    subjects: Optional[List[str]] = None,
    split: str = "test",
) -> List[MMLUExample]:
    """Load MMLU-Pro examples from local disk.

    MMLU-Pro has 10 answer choices (A-J) instead of 4. The ``subject`` field
    on returned examples corresponds to the ``category`` column in the dataset.

    Args:
        n_samples: Number of examples to return. None for all.
        seed: Random seed for even-across-subjects sampling.
        data_dir: Local path to MMLU-Pro dataset saved with save_to_disk().
        subjects: List of categories to include (default: all). E.g. ``["math", "physics"]``.
        split: Dataset split to load (default: "test").

    Returns:
        List of MMLUExample instances (choices may have up to 10 items).
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets library is required. "
            "Install with: pip install datasets"
        )

    from datasets import load_from_disk, load_dataset as _load_dataset  # pylint: disable=import-outside-toplevel

    local_path = Path(data_dir)
    if not local_path.exists():
        raise FileNotFoundError(
            f"Local MMLU-Pro dataset not found at '{data_dir}'. "
            "Download it once on a login node with:\n"
            "  from datasets import load_dataset\n"
            "  ds = load_dataset('TIGER-Lab/MMLU-Pro', split='test')\n"
            f"  ds.save_to_disk('{data_dir}')"
        )

    try:
        dataset = load_from_disk(data_dir)
        if hasattr(dataset, "__getitem__") and split in dataset:
            dataset = dataset[split]
    except Exception:  # pylint: disable=broad-exception-caught
        dataset = _load_dataset(data_dir, split=split)

    subject_counters: Dict[str, int] = {}
    entries: List[MMLUExample] = []
    for item in dataset:
        subject = item["category"]
        if subjects is not None and subject not in subjects:
            continue
        idx = subject_counters.get(subject, 0)
        subject_counters[subject] = idx + 1
        entries.append(MMLUExample(
            id=f"mmlu_pro_{subject}_{idx:04d}",
            question=item["question"],
            choices=list(item["options"]),
            expected=item["answer"],  # already a letter, e.g. "A"–"J"
            subject=subject,
        ))

    if n_samples is not None and len(entries) > n_samples:
        random.seed(seed)
        by_subject: Dict[str, List] = {}
        for e in entries:
            by_subject.setdefault(e.subject, []).append(e)

        subject_list = sorted(by_subject.keys())
        per_subject = max(1, n_samples // len(subject_list))
        sampled: List[MMLUExample] = []
        for subj in subject_list:
            pool = by_subject[subj]
            sampled.extend(random.sample(pool, min(per_subject, len(pool))))

        if len(sampled) < n_samples:
            remaining = [e for e in entries if e not in sampled]
            sampled.extend(random.sample(remaining, n_samples - len(sampled)))

        entries = sampled[:n_samples]

    return entries


@dataclass
class MATHExample:
    """Single Hendrycks MATH competition example."""

    id: str
    question: str
    answer: str   # final answer, extracted from \boxed{...}
    subject: str  # e.g. "Algebra", "Number Theory"
    level: int    # difficulty 1–5


def _extract_boxed(solution: str) -> str:
    """Extract the content of the last \\boxed{...} in a solution string."""
    # Handle nested braces by counting depth
    tag = r"\boxed{"
    idx = solution.rfind(tag)
    if idx == -1:
        return solution.strip()
    start = idx + len(tag)
    depth = 1
    for i, ch in enumerate(solution[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return solution[start:i].strip()
    return solution[start:].strip()


def load_math_subset(
    n_samples: Optional[int] = 100,
    seed: int = 42,
    data_dir: str = _DEFAULT_MATH_DATA_DIR,
    subjects: Optional[List[str]] = None,
    levels: Optional[List[int]] = None,
    split: str = "test",
) -> List[MATHExample]:
    """Load Hendrycks MATH examples from local disk.

    Args:
        n_samples: Number of examples to return. None for all.
        seed: Random seed for even-across-subjects sampling.
        data_dir: Local path to MATH dataset saved with save_to_disk().
        subjects: List of subjects to include (default: all 7). E.g. ``["Algebra", "Geometry"]``.
        levels: List of difficulty levels to include, 1–5 (default: all).
        split: Dataset split to load (default: "test").

    Returns:
        List of MATHExample instances.
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets library is required. "
            "Install with: pip install datasets"
        )

    from datasets import load_from_disk, load_dataset as _load_dataset  # pylint: disable=import-outside-toplevel

    local_path = Path(data_dir)
    if not local_path.exists():
        raise FileNotFoundError(
            f"Local MATH dataset not found at '{data_dir}'. "
            "Download it once on a login node with:\n"
            "  from datasets import load_dataset\n"
            "  ds = load_dataset('qwedsacf/competition_math', split='test')\n"
            f"  ds.save_to_disk('{data_dir}')"
        )

    try:
        dataset = load_from_disk(data_dir)
        if hasattr(dataset, "__getitem__") and split in dataset:
            dataset = dataset[split]
    except Exception:  # pylint: disable=broad-exception-caught
        dataset = _load_dataset(data_dir, split=split)

    subject_counters: Dict[str, int] = {}
    entries: List[MATHExample] = []
    for item in dataset:
        # Field names vary slightly across HF versions of this dataset
        subject = item.get("type") or item.get("subject", "unknown")
        level_str = item.get("level", "Level 1")
        try:
            level = int(re.search(r"\d", level_str).group())
        except (AttributeError, ValueError):
            level = 0

        if subjects is not None and subject not in subjects:
            continue
        if levels is not None and level not in levels:
            continue

        solution = item.get("solution", "")
        answer = item.get("answer") or _extract_boxed(solution)

        idx = subject_counters.get(subject, 0)
        subject_counters[subject] = idx + 1
        entries.append(MATHExample(
            id=f"math_{subject.replace(' ', '_')}_{idx:04d}",
            question=item["problem"],
            answer=answer,
            subject=subject,
            level=level,
        ))

    if n_samples is not None and len(entries) > n_samples:
        random.seed(seed)
        by_subject: Dict[str, List] = {}
        for e in entries:
            by_subject.setdefault(e.subject, []).append(e)

        subject_list = sorted(by_subject.keys())
        per_subject = max(1, n_samples // len(subject_list))
        sampled: List[MATHExample] = []
        for subj in subject_list:
            pool = by_subject[subj]
            sampled.extend(random.sample(pool, min(per_subject, len(pool))))

        if len(sampled) < n_samples:
            remaining = [e for e in entries if e not in sampled]
            sampled.extend(random.sample(remaining, n_samples - len(sampled)))

        entries = sampled[:n_samples]

    return entries


# =============================================================================
# SVAMP Dataset (Simple Variations on Arithmetic Math word Problems)
# =============================================================================

_DEFAULT_SVAMP_DATA_DIR = os.environ.get("DATASETS_DIR") and os.path.join(os.environ["DATASETS_DIR"], "svamp")


def load_svamp_subset(
    n_samples: Optional[int] = 100,
    seed: int = 42,
    data_dir: str = _DEFAULT_SVAMP_DATA_DIR,
    split: str = "test",
) -> List[ReasoningExample]:
    """Load SVAMP examples from local disk.

    SVAMP (Simple Variations on Arithmetic Math word Problems) is a ~700-example
    test-only dataset. Each problem has a body, a question, and a numerical answer.

    Args:
        n_samples: Number of examples to return. None for all.
        seed: Random seed for sampling.
        data_dir: Local path to SVAMP dataset saved with save_to_disk().
        split: Dataset split to load (default: "test").

    Returns:
        List of ReasoningExample instances.

    Download once on a login node with::

        from datasets import load_dataset
        ds = load_dataset('ChilleD/SVAMP', split='test')
        ds.save_to_disk('${DATASETS_DIR}/svamp')
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets library is required. "
            "Install with: pip install datasets"
        )

    from datasets import load_from_disk, load_dataset as _load_dataset  # pylint: disable=import-outside-toplevel

    local_path = Path(data_dir)
    if not local_path.exists():
        raise FileNotFoundError(
            f"Local SVAMP dataset not found at '{data_dir}'. "
            "Download it once on a login node with:\n"
            "  from datasets import load_dataset\n"
            "  ds = load_dataset('ChilleD/SVAMP', split='test')\n"
            f"  ds.save_to_disk('{data_dir}')"
        )

    try:
        dataset = load_from_disk(data_dir)
        if hasattr(dataset, "__getitem__") and split in dataset:
            dataset = dataset[split]
    except Exception:  # pylint: disable=broad-exception-caught
        dataset = _load_dataset(data_dir, split=split)

    entries: List[ReasoningExample] = []
    for i, item in enumerate(dataset):
        body = item.get("Body", "").strip()
        question = item.get("Question", "").strip()
        full_question = f"{body} {question}".strip()

        raw_answer = item.get("Answer", 0)
        try:
            num = float(raw_answer)
        except (ValueError, TypeError):
            num = None

        # Store as integer string when possible (e.g. "42" not "42.0")
        if num is not None and num == int(num):
            answer_str = str(int(num))
        else:
            answer_str = str(num) if num is not None else str(raw_answer)

        entries.append(ReasoningExample(
            id=f"svamp_{i:04d}",
            question=full_question,
            answer=answer_str,
            numerical_answer=num,
        ))

    if n_samples is not None and len(entries) > n_samples:
        random.seed(seed)
        entries = random.sample(entries, n_samples)

    return entries


# =============================================================================
# GSM-Plus Dataset
# =============================================================================

_DEFAULT_GSM_PLUS_DATA_DIR = os.environ.get("DATASETS_DIR") and os.path.join(os.environ["DATASETS_DIR"], "gsm_plus")


@dataclass
class GSMPlusExample:
    """Single GSM-Plus example (GSM8K variant with perturbation metadata)."""

    id: str
    question: str
    answer: str                   # numerical answer as string
    perturbation_type: str
    chain_of_thought: Optional[str] = None
    numerical_answer: Optional[float] = None


def load_gsm_plus_subset(
    n_samples: Optional[int] = 100,
    seed: int = 42,
    data_dir: str = _DEFAULT_GSM_PLUS_DATA_DIR,
    perturbation_types: Optional[List[str]] = None,
    split: str = "test",
) -> List[GSMPlusExample]:
    """Load GSM-Plus examples from local disk.

    GSM-Plus augments GSM8K with systematic perturbations such as numerical
    substitution, digit expansion, distractor insertion, etc.  The answer is
    stored in ``solution`` and follows the same ``#### <number>`` convention
    as GSM8K.

    Args:
        n_samples: Number of examples to return. None for all.
        seed: Random seed for sampling.
        data_dir: Local path to GSM-Plus dataset saved with save_to_disk().
        perturbation_types: Perturbation types to include (default: all).
            E.g. ``["numerical substitution", "digit expansion"]``.
        split: Dataset split to load (default: "test").

    Returns:
        List of GSMPlusExample instances.

    Download once on a login node with::

        from datasets import load_dataset
        ds = load_dataset('qintongli/GSM-Plus', split='test')
        ds.save_to_disk('${DATASETS_DIR}/gsm_plus')
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets library is required. "
            "Install with: pip install datasets"
        )

    from datasets import load_from_disk, load_dataset as _load_dataset  # pylint: disable=import-outside-toplevel

    local_path = Path(data_dir)
    if not local_path.exists():
        raise FileNotFoundError(
            f"Local GSM-Plus dataset not found at '{data_dir}'. "
            "Download it once on a login node with:\n"
            "  from datasets import load_dataset\n"
            "  ds = load_dataset('qintongli/GSM-Plus', split='test')\n"
            f"  ds.save_to_disk('{data_dir}')"
        )

    try:
        dataset = load_from_disk(data_dir)
        if hasattr(dataset, "__getitem__") and split in dataset:
            dataset = dataset[split]
    except Exception:  # pylint: disable=broad-exception-caught
        dataset = _load_dataset(data_dir, split=split)

    entries: List[GSMPlusExample] = []
    for i, item in enumerate(dataset):
        p_type = item.get("perturbation_type", "unknown")
        if perturbation_types is not None and p_type not in perturbation_types:
            continue

        solution = item.get("solution", "")
        numerical = extract_numerical_answer(solution)
        cot = solution.split("####")[0].strip() if "####" in solution else solution

        if numerical is not None and numerical == int(numerical):
            answer_str = str(int(numerical))
        else:
            answer_str = str(numerical) if numerical is not None else solution

        entries.append(GSMPlusExample(
            id=f"gsm_plus_{i:04d}",
            question=item.get("question", ""),
            answer=answer_str,
            perturbation_type=p_type,
            chain_of_thought=cot,
            numerical_answer=numerical,
        ))

    if n_samples is not None and len(entries) > n_samples:
        random.seed(seed)
        entries = random.sample(entries, n_samples)

    return entries


# =============================================================================
# GSM-Symbolic Dataset (Apple, p1 template)
# =============================================================================

_DEFAULT_GSM_SYMBOLIC_DATA_DIR = os.environ.get("DATASETS_DIR") and os.path.join(os.environ["DATASETS_DIR"], "gsm_symbolic")


def load_gsm_symbolic_subset(
    n_samples: Optional[int] = 100,
    seed: int = 42,
    data_dir: str = _DEFAULT_GSM_SYMBOLIC_DATA_DIR,
    split: str = "test",
) -> List[ReasoningExample]:
    """Load GSM-Symbolic (p1 template) examples from local disk.

    GSM-Symbolic is Apple's symbolic variant of GSM8K (p1 adds one extra
    reasoning step).  The answer field follows the same ``#### <number>``
    convention as GSM8K.

    Args:
        n_samples: Number of examples to return. None for all.
        seed: Random seed for sampling.
        data_dir: Local path to GSM-Symbolic p1 dataset saved with save_to_disk().
        split: Dataset split to load (default: "test").

    Returns:
        List of ReasoningExample instances.

    Download once on a login node with::

        from datasets import load_dataset
        ds = load_dataset('apple/GSM-Symbolic', 'p1', split='test')
        ds.save_to_disk('${DATASETS_DIR}/gsm_symbolic')
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets library is required. "
            "Install with: pip install datasets"
        )

    from datasets import load_from_disk, load_dataset as _load_dataset  # pylint: disable=import-outside-toplevel

    local_path = Path(data_dir)
    if not local_path.exists():
        raise FileNotFoundError(
            f"Local GSM-Symbolic dataset not found at '{data_dir}'. "
            "Download it once on a login node with:\n"
            "  from datasets import load_dataset\n"
            "  ds = load_dataset('apple/GSM-Symbolic', 'p1', split='test')\n"
            f"  ds.save_to_disk('{data_dir}')"
        )

    try:
        dataset = load_from_disk(data_dir)
        if hasattr(dataset, "__getitem__") and split in dataset:
            dataset = dataset[split]
    except Exception:  # pylint: disable=broad-exception-caught
        dataset = _load_dataset(data_dir, split=split)

    entries: List[ReasoningExample] = []
    for i, item in enumerate(dataset):
        answer_text = item.get("answer", "")
        numerical = extract_numerical_answer(answer_text)
        cot = answer_text.split("####")[0].strip() if "####" in answer_text else answer_text

        if numerical is not None and numerical == int(numerical):
            answer_str = str(int(numerical))
        else:
            answer_str = str(numerical) if numerical is not None else answer_text

        entries.append(ReasoningExample(
            id=f"gsm_symbolic_{i:04d}",
            question=item.get("question", ""),
            answer=answer_str,
            chain_of_thought=cot,
            numerical_answer=numerical,
        ))

    if n_samples is not None and len(entries) > n_samples:
        random.seed(seed)
        entries = random.sample(entries, n_samples)

    return entries


# =============================================================================
# CodeMMLU Dataset (Fsoft-AIC/CodeMMLU)
# =============================================================================

_DEFAULT_CODEMMLU_DATA_DIR = os.environ.get("DATASETS_DIR") and os.path.join(os.environ["DATASETS_DIR"], "codemmlu")

_CODEMMLU_CONFIGS = [
    "api_frameworks",
    "code_completion",
    "code_repair",
    "dbms_sql",
    "execution_prediction",
    "fill_in_the_middle",
    "others",
    "programming_syntax",
    "software_principles",
]

_CODEMMLU_CHOICE_LABELS = ["A", "B", "C", "D"]


def load_codemmlu_subset(
    n_samples: Optional[int] = 100,
    seed: int = 42,
    data_dir: str = _DEFAULT_CODEMMLU_DATA_DIR,
    subjects: Optional[List[str]] = None,
) -> List[MMLUExample]:
    """Load CodeMMLU examples from local disk.

    CodeMMLU (Fsoft-AIC/CodeMMLU) is a multiple-choice benchmark covering
    code understanding across 9 categories: api_frameworks, code_completion,
    code_repair, dbms_sql, execution_prediction, fill_in_the_middle, others,
    programming_syntax, software_principles.

    The dataset has only a test split.  ``MMLUExample`` is reused since the
    structure is identical (question, choices, letter answer, subject).

    Args:
        n_samples: Number of examples to return. None for all.
        seed: Random seed for sampling.
        data_dir: Local path saved with save_to_disk() (see download command below).
        subjects: Subset of config names to include (default: all 9).

    Returns:
        List of MMLUExample instances.

    Download once on a login node with::

        from datasets import load_dataset, Dataset
        configs = [
            "api_frameworks", "code_completion", "code_repair", "dbms_sql",
            "execution_prediction", "fill_in_the_middle", "others",
            "programming_syntax", "software_principles",
        ]
        rows = []
        for cfg in configs:
            for item in load_dataset("Fsoft-AIC/CodeMMLU", cfg, split="test"):
                item["subject"] = cfg
                rows.append(item)
        Dataset.from_list(rows).save_to_disk(
            "${DATASETS_DIR}/codemmlu"
        )
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets library is required. "
            "Install with: pip install datasets"
        )

    from datasets import load_from_disk, load_dataset as _load_dataset  # pylint: disable=import-outside-toplevel

    local_path = Path(data_dir)
    if not local_path.exists():
        raise FileNotFoundError(
            f"Local CodeMMLU dataset not found at '{data_dir}'. "
            "Download it once on a login node — see the docstring for instructions."
        )

    try:
        dataset = load_from_disk(data_dir)
    except Exception:  # pylint: disable=broad-exception-caught
        dataset = _load_dataset(data_dir, split="test")

    subject_counters: Dict[str, int] = {}
    entries: List[MMLUExample] = []
    for item in dataset:
        subject = item.get("subject", "unknown")
        if subjects is not None and subject not in subjects:
            continue
        choices = list(item["choices"])
        # Normalise to 4 choices (pad with empty strings if fewer)
        while len(choices) < 4:
            choices.append("")
        choices = choices[:4]

        answer = str(item["answer"]).strip().upper()
        if answer not in _CODEMMLU_CHOICE_LABELS:
            continue  # skip malformed rows

        idx = subject_counters.get(subject, 0)
        subject_counters[subject] = idx + 1
        entries.append(MMLUExample(
            id=f"codemmlu_{subject}_{idx:04d}",
            question=item["question"],
            choices=choices,
            expected=answer,
            subject=subject,
        ))

    if n_samples is not None and len(entries) > n_samples:
        random.seed(seed)
        by_subject: Dict[str, List] = {}
        for e in entries:
            by_subject.setdefault(e.subject, []).append(e)

        subject_list = sorted(by_subject.keys())
        per_subject = max(1, n_samples // len(subject_list))
        sampled: List[MMLUExample] = []
        for subj in subject_list:
            pool = by_subject[subj]
            sampled.extend(random.sample(pool, min(per_subject, len(pool))))

        if len(sampled) < n_samples:
            remaining = [e for e in entries if e not in sampled]
            sampled.extend(random.sample(remaining, n_samples - len(sampled)))

        entries = sampled[:n_samples]

    return entries


# =============================================================================
# RBI Circular QA Dataset (Vishva007/RBI-Circular-QA-Dataset)
# =============================================================================

_DEFAULT_RBI_DATA_DIR = os.environ.get("DATASETS_DIR") and os.path.join(os.environ["DATASETS_DIR"], "rbi")


def load_rbi_subset(
    n_samples: Optional[int] = 100,
    seed: int = 42,
    data_dir: str = _DEFAULT_RBI_DATA_DIR,
    split: str = "eval",
) -> List[ReasoningExample]:
    """
    Load RBI Circular QA dataset.

    This is an open-ended QA benchmark over Reserve Bank of India regulatory
    circulars.  Each example has a free-form question and a reference answer.
    Evaluation uses token-level F1 score (threshold >= 0.3 counts as correct).

    Args:
        n_samples: Number of samples.  None for all.
        seed: Random seed.
        data_dir: Path to dataset saved with ``save_to_disk()``.
        split: ``"train"`` or ``"eval"``.  Default ``"eval"`` (1 000 examples).

    Returns:
        List of ReasoningExample instances.
    """
    from datasets import load_from_disk  # pylint: disable=import-outside-toplevel

    ds = load_from_disk(data_dir)
    data = ds[split]

    entries: List[ReasoningExample] = []
    for idx, item in enumerate(data):
        entries.append(ReasoningExample(
            id=f"rbi_{split}_{idx:05d}",
            question=item["question"],
            answer=item["answer"],
        ))

    if n_samples is not None and len(entries) > n_samples:
        random.seed(seed)
        entries = random.sample(entries, n_samples)

    return entries


# =============================================================================
# MedMCQA Dataset (openlifescienceai/medmcqa)
# =============================================================================

_DEFAULT_MEDMCQA_DATA_DIR = os.environ.get("DATASETS_DIR") and os.path.join(os.environ["DATASETS_DIR"], "medmcqa")

_MEDMCQA_CHOICE_LABELS = ["A", "B", "C", "D"]


def load_medmcqa_subset(
    n_samples: Optional[int] = 100,
    seed: int = 42,
    data_dir: str = _DEFAULT_MEDMCQA_DATA_DIR,
    split: str = "validation",
) -> List[MMLUExample]:
    """
    Load MedMCQA dataset (single-choice questions only).

    ``cop`` is 0-indexed (0=A, 1=B, 2=C, 3=D).  Only ``choice_type == "single"``
    examples are included.  ``MMLUExample`` is reused; ``subject`` = subject_name.

    Args:
        n_samples: Number of samples.  None for all.
        seed: Random seed.
        data_dir: Path to dataset saved with ``save_to_disk()``.
        split: ``"train"``, ``"validation"``, or ``"test"``.
            Default ``"validation"`` (4 183 examples, 2 816 single-choice).

    Returns:
        List of MMLUExample instances.
    """
    from datasets import load_from_disk  # pylint: disable=import-outside-toplevel

    ds = load_from_disk(data_dir)
    data = ds[split]

    entries: List[MMLUExample] = []
    for idx, item in enumerate(data):
        if item["choice_type"] != "single":
            continue
        choices = [item["opa"], item["opb"], item["opc"], item["opd"]]
        expected = _MEDMCQA_CHOICE_LABELS[int(item["cop"])]
        entries.append(MMLUExample(
            id=f"medmcqa_{split}_{idx:06d}",
            question=item["question"],
            choices=choices,
            expected=expected,
            subject=item["subject_name"] or "unknown",
        ))

    if n_samples is not None and len(entries) > n_samples:
        random.seed(seed)
        entries = random.sample(entries, n_samples)

    return entries


if __name__ == "__main__":
    # Test dataset creation
    print("Creating reasoning datasets...\n")

    # GSM8k from HuggingFace (preferred)
    print("=" * 60)
    print("GSM8K from HuggingFace")
    print("=" * 60)
    if HF_DATASETS_AVAILABLE:
        try:
            gsm8k_hf = load_gsm8k_from_huggingface(n_samples=10, split="test")
            print(f"GSM8k (HuggingFace): {len(gsm8k_hf)} examples")
            print(f"  Example: {gsm8k_hf[0].question[:60]}...")
            print(f"  Answer: {gsm8k_hf[0].answer}")
            print(f"  Chain of thought: {gsm8k_hf[0].chain_of_thought[:80]}...")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Failed to load from HuggingFace: {e}")
    else:
        print("HuggingFace datasets not available")
    print()

    # GSM8k subset (will try HF first, then fall back to sample)
    print("=" * 60)
    print("GSM8K via load_gsm8k_subset()")
    print("=" * 60)
    gsm8k = load_gsm8k_subset(n_samples=10)
    print(f"GSM8k: {len(gsm8k)} examples")
    print(f"  Example: {gsm8k[0].question[:60]}...")
    print(f"  Answer: {gsm8k[0].answer}")
    print()


# =============================================================================
# WikiMIA Dataset
# =============================================================================

@dataclass
class WikiMIAExample:
    """Single WikiMIA example (Wikipedia sentence with seen/unseen label)."""

    id: str
    text: str          # the Wikipedia sentence to score
    label: int         # 1 = seen (pre-2023), 0 = unseen (post-2023)
    question: str = "" # unused; kept for interface compatibility
    answer: str = ""   # set to text in __post_init__

    def __post_init__(self):
        self.answer = self.text


def load_wikimia_subset(
    split: str = "WikiMIA_length64",
    n_samples: Optional[int] = None,
    seed: int = 42,
    data_dir: Optional[str] = None,
) -> List[WikiMIAExample]:
    """
    Load WikiMIA dataset (swj0419/WikiMIA).

    Each example has a Wikipedia sentence and a binary label:
      1 = pre-2023 (seen by models released before 2023)
      0 = post-2023 (unseen)

    Available splits: WikiMIA_length32, WikiMIA_length64, WikiMIA_length128, WikiMIA_length256.

    Args:
        split: Dataset split name (default: WikiMIA_length64).
        n_samples: Number of examples to return. None = all.
        seed: Random seed for sampling.
        data_dir: Local path to saved dataset. If None, tries $SCRATCH/data/wikimia_length*.

    Returns:
        List of WikiMIAExample.
    """
    length_tag = split.replace("WikiMIA_length", "")
    examples = []

    # Try local disk first
    if data_dir is None:
        scratch = os.environ.get("DATASETS_DIR") or os.environ.get("SCRATCH", "")
        if not scratch:
            raise ValueError(
                "Set DATASETS_DIR (or SCRATCH) or pass data_dir explicitly."
            )
        data_dir = os.path.join(scratch, f"data/wikimia_length{length_tag}")

    local_path = Path(data_dir)
    if local_path.exists():
        from datasets import load_from_disk  # pylint: disable=import-outside-toplevel
        ds = load_from_disk(str(local_path))
    elif HF_DATASETS_AVAILABLE:
        ds = load_dataset("swj0419/WikiMIA", split=split)
    else:
        raise RuntimeError(
            f"WikiMIA not found at {data_dir} and HuggingFace datasets not available."
        )

    for i, row in enumerate(ds):
        examples.append(WikiMIAExample(
            id=f"wikimia_{split}_{i}",
            text=row["input"],
            label=int(row["label"]),
        ))

    if n_samples is not None and n_samples < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, n_samples)

    return examples


# =============================================================================
# NuminaMath-TIR Dataset
# =============================================================================

_DEFAULT_NUMINAMATH_TIR_DATA_DIR = os.environ.get("DATASETS_DIR") and os.path.join(os.environ["DATASETS_DIR"], "numinamath_tir")


def load_numinamath_tir_subset(
    n_samples: Optional[int] = None,
    seed: int = 42,
    data_dir: str = _DEFAULT_NUMINAMATH_TIR_DATA_DIR,
    split: str = "train",
) -> List[ReasoningExample]:
    """Load NuminaMath-TIR examples from local disk (AI-MO/NuminaMath-TIR).

    Each example has a 'problem' field and a 'solution' field (TIR-format with
    Python code blocks).  For --input-only memorization detection, the answer
    is set to the full solution so that the forward pass scores P(problem +
    solution).

    Args:
        n_samples: Number of examples to return. None = all.
        seed: Random seed for sampling.
        data_dir: Local path to dataset saved with save_to_disk().
        split: Dataset split to load ('train' or 'test'). Test has ~99 examples.

    Returns:
        List of ReasoningExample instances.
    """
    from datasets import load_from_disk  # pylint: disable=import-outside-toplevel

    local_path = Path(data_dir)
    if not local_path.exists():
        raise FileNotFoundError(
            f"NuminaMath-TIR dataset not found at '{data_dir}'. "
            "Download with:\n"
            "  from datasets import load_dataset\n"
            "  ds = load_dataset('AI-MO/NuminaMath-TIR')\n"
            f"  ds.save_to_disk('{data_dir}')"
        )

    ds = load_from_disk(str(local_path))
    if hasattr(ds, "keys") and split in ds:
        ds = ds[split]

    examples: List[ReasoningExample] = []
    for i, row in enumerate(ds):
        problem = row.get("problem", row.get("question", "")).strip()
        solution = row.get("solution", row.get("answer", "")).strip()
        examples.append(ReasoningExample(
            id=f"numinamath_tir_{split}_{i}",
            question=problem,
            answer=solution,
        ))

    if n_samples is not None and n_samples < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, n_samples)

    return examples
