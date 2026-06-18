"""Tokenizer alignment utilities for cross-model grafting."""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from transformers import PreTrainedTokenizer


@dataclass
class TokenAlignment:
    """Alignment information between two tokenizations of the same text."""

    text: str
    tokens_a: List[str]
    tokens_b: List[str]
    token_ids_a: List[int]
    token_ids_b: List[int]
    # Mapping from A's token indices to B's token indices (many-to-many possible)
    a_to_b: List[List[int]]
    b_to_a: List[List[int]]
    # Whether final tokens align to the same character span
    is_compatible: bool
    # Character spans for final tokens
    final_span_a: Tuple[int, int]
    final_span_b: Tuple[int, int]


class TokenizerAligner:
    """
    Utilities for handling tokenizer mismatches between models.

    When grafting activations between models with different tokenizers,
    the "final token" may correspond to different subword units. This
    class provides alignment utilities to handle such cases.

    Example:
        Model A tokenizes "France" -> ["Fr", "ance"]
        Model B tokenizes "France" -> ["France"]

        The final token positions don't align, which can affect grafting.
    """

    def __init__(
        self,
        tokenizer_a: PreTrainedTokenizer,
        tokenizer_b: PreTrainedTokenizer,
    ):
        """
        Initialize the aligner.

        Args:
            tokenizer_a: Tokenizer for Model A
            tokenizer_b: Tokenizer for Model B
        """
        self.tokenizer_a = tokenizer_a
        self.tokenizer_b = tokenizer_b
        self._same_tokenizer = self._check_same_tokenizer()

    def _check_same_tokenizer(self) -> bool:
        """Check if tokenizers are functionally identical."""
        # Check by name
        name_a = getattr(self.tokenizer_a, "name_or_path", "")
        name_b = getattr(self.tokenizer_b, "name_or_path", "")

        if name_a and name_b and name_a == name_b:
            return True

        # Check by vocab size and a sample
        if self.tokenizer_a.vocab_size != self.tokenizer_b.vocab_size:
            return False

        # Test on sample text
        test_texts = ["Hello world", "The quick brown fox"]
        for text in test_texts:
            ids_a = self.tokenizer_a.encode(text)
            ids_b = self.tokenizer_b.encode(text)
            if ids_a != ids_b:
                return False

        return True

    @property
    def is_identical(self) -> bool:
        """True if tokenizers are identical (no alignment needed)."""
        return self._same_tokenizer

    def align(self, text: str) -> TokenAlignment:  # pylint: disable=too-many-branches
        """
        Compute token alignment between the two tokenizers.

        Uses character offset mapping to align tokens.

        Args:
            text: Text to tokenize and align

        Returns:
            TokenAlignment with mapping information
        """
        # Get tokens with offsets
        enc_a = self.tokenizer_a(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        enc_b = self.tokenizer_b(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        offsets_a = enc_a.get("offset_mapping", [])
        offsets_b = enc_b.get("offset_mapping", [])

        token_ids_a = enc_a["input_ids"]
        token_ids_b = enc_b["input_ids"]

        tokens_a = self.tokenizer_a.convert_ids_to_tokens(token_ids_a)
        tokens_b = self.tokenizer_b.convert_ids_to_tokens(token_ids_b)

        # Handle case where offset mapping is not available
        if not offsets_a or not offsets_b:
            # Fall back to simple index-based alignment
            return TokenAlignment(
                text=text,
                tokens_a=tokens_a,
                tokens_b=tokens_b,
                token_ids_a=token_ids_a,
                token_ids_b=token_ids_b,
                a_to_b=[[i] if i < len(tokens_b) else [] for i in range(len(tokens_a))],
                b_to_a=[[i] if i < len(tokens_a) else [] for i in range(len(tokens_b))],
                is_compatible=len(tokens_a) == len(tokens_b),
                final_span_a=(0, len(text)),
                final_span_b=(0, len(text)),
            )

        # Build alignment via character overlap
        a_to_b = []
        for i, (start_a, end_a) in enumerate(offsets_a):
            if start_a is None or end_a is None:
                a_to_b.append([])
                continue
            aligned = []
            for j, (start_b, end_b) in enumerate(offsets_b):
                if start_b is None or end_b is None:
                    continue
                # Check for character span overlap
                if start_a < end_b and start_b < end_a:
                    aligned.append(j)
            a_to_b.append(aligned)

        b_to_a = []
        for j, (start_b, end_b) in enumerate(offsets_b):
            if start_b is None or end_b is None:
                b_to_a.append([])
                continue
            aligned = []
            for i, (start_a, end_a) in enumerate(offsets_a):
                if start_a is None or end_a is None:
                    continue
                if start_a < end_b and start_b < end_a:
                    aligned.append(i)
            b_to_a.append(aligned)

        # Check if final tokens align
        final_a = offsets_a[-1] if offsets_a else (0, 0)
        final_b = offsets_b[-1] if offsets_b else (0, 0)

        # Consider compatible if final tokens overlap significantly
        if final_a[0] is not None and final_b[0] is not None:
            overlap_start = max(final_a[0], final_b[0])
            overlap_end = min(final_a[1], final_b[1])
            overlap = max(0, overlap_end - overlap_start)
            span_a = final_a[1] - final_a[0]
            span_b = final_b[1] - final_b[0]
            # Compatible if >50% overlap with either
            is_compatible = (
                overlap > 0.5 * min(span_a, span_b)
                if min(span_a, span_b) > 0
                else False
            )
        else:
            is_compatible = False

        return TokenAlignment(
            text=text,
            tokens_a=tokens_a,
            tokens_b=tokens_b,
            token_ids_a=token_ids_a,
            token_ids_b=token_ids_b,
            a_to_b=a_to_b,
            b_to_a=b_to_a,
            is_compatible=is_compatible,
            final_span_a=final_a if final_a[0] is not None else (0, len(text)),
            final_span_b=final_b if final_b[0] is not None else (0, len(text)),
        )

    def get_final_token_positions(
        self,
        text: str,
    ) -> Tuple[int, int, bool]:
        """
        Get final token positions and alignment status.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (pos_a, pos_b, aligned) where:
                pos_a: Final token position for tokenizer A
                pos_b: Final token position for tokenizer B
                aligned: Whether they cover the same character span
        """
        alignment = self.align(text)
        pos_a = len(alignment.tokens_a) - 1
        pos_b = len(alignment.tokens_b) - 1
        return pos_a, pos_b, alignment.is_compatible

    def get_aligned_positions(
        self,
        text: str,
        position_a: int,
    ) -> List[int]:
        """
        Get positions in B that align with a position in A.

        Args:
            text: Text that was tokenized
            position_a: Token position in A's tokenization

        Returns:
            List of aligned positions in B's tokenization
        """
        alignment = self.align(text)
        if position_a < 0:
            position_a = len(alignment.tokens_a) + position_a
        if 0 <= position_a < len(alignment.a_to_b):
            return alignment.a_to_b[position_a]
        return []

    def audit_compatibility(
        self,
        test_texts: List[str],
    ) -> Dict[str, Any]:
        """
        Audit tokenizer compatibility over a set of test texts.

        Args:
            test_texts: List of texts to test

        Returns:
            Dictionary with compatibility statistics
        """
        results = {
            "total": len(test_texts),
            "identical_tokenization": 0,
            "final_token_aligned": 0,
            "length_ratios": [],
            "misaligned_examples": [],
        }

        for text in test_texts:
            alignment = self.align(text)
            len_a = len(alignment.tokens_a)
            len_b = len(alignment.tokens_b)

            if len_b > 0:
                results["length_ratios"].append(len_a / len_b)

            if alignment.tokens_a == alignment.tokens_b:
                results["identical_tokenization"] += 1

            if alignment.is_compatible:
                results["final_token_aligned"] += 1
            else:
                if len(results["misaligned_examples"]) < 10:
                    results["misaligned_examples"].append({
                        "text": text[:100] + ("..." if len(text) > 100 else ""),
                        "tokens_a_final": alignment.tokens_a[-3:] if alignment.tokens_a else [],
                        "tokens_b_final": alignment.tokens_b[-3:] if alignment.tokens_b else [],
                        "span_a": alignment.final_span_a,
                        "span_b": alignment.final_span_b,
                    })

        # Compute summary statistics
        if results["length_ratios"]:
            results["length_ratio_mean"] = (
                sum(results["length_ratios"]) / len(results["length_ratios"])
            )
            variance = sum(
                (r - results["length_ratio_mean"]) ** 2
                for r in results["length_ratios"]
            ) / len(results["length_ratios"])
            results["length_ratio_std"] = variance ** 0.5
        else:
            results["length_ratio_mean"] = 1.0
            results["length_ratio_std"] = 0.0

        results["identical_pct"] = (
            results["identical_tokenization"] / results["total"]
            if results["total"] > 0
            else 0
        )
        results["aligned_pct"] = (
            results["final_token_aligned"] / results["total"]
            if results["total"] > 0
            else 0
        )

        # Clean up for return
        del results["length_ratios"]

        return results


def check_tokenizer_compatibility(
    tokenizer_a: PreTrainedTokenizer,
    tokenizer_b: PreTrainedTokenizer,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Quick compatibility check between two tokenizers.

    Args:
        tokenizer_a: First tokenizer
        tokenizer_b: Second tokenizer
        verbose: Whether to print results

    Returns:
        Compatibility information
    """
    aligner = TokenizerAligner(tokenizer_a, tokenizer_b)

    # Test texts covering various cases
    test_texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "Python programming language",
        "The answer is 42.",
        "Let me think step by step...",
        "José García lives in São Paulo",  # Unicode
        "```python\nprint('hello')\n```",  # Code
        "The Eiffel Tower is located in France.",
        "What is 2 + 2?",
    ]

    results = aligner.audit_compatibility(test_texts)
    results["is_identical"] = aligner.is_identical

    if verbose:
        print("Tokenizer Compatibility Report")
        print("=" * 50)
        print(f"  Identical tokenizers: {results['is_identical']}")
        print(f"  Identical tokenization: {results['identical_pct']*100:.1f}%")
        print(f"  Final token aligned: {results['aligned_pct']*100:.1f}%")
        print(
            f"  Length ratio (A/B): {results['length_ratio_mean']:.3f}"
            f" ± {results['length_ratio_std']:.3f}"
        )

        if results["misaligned_examples"]:
            print(f"\n  Misaligned examples ({len(results['misaligned_examples'])}):")
            for ex in results["misaligned_examples"][:3]:
                print(f"    Text: {ex['text'][:50]}...")
                print(f"    A final tokens: {ex['tokens_a_final']}")
                print(f"    B final tokens: {ex['tokens_b_final']}")

    return results


if __name__ == "__main__":
    from transformers import AutoTokenizer

    print("Testing tokenizer alignment...")

    # Same family
    print("\n" + "=" * 60)
    print("Same family (Qwen 0.5B vs 1.5B):")
    print("=" * 60)
    tok_a = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    tok_b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    check_tokenizer_compatibility(tok_a, tok_b)

    # Cross family
    print("\n" + "=" * 60)
    print("Cross family (Qwen vs TinyLlama):")
    print("=" * 60)
    tok_a = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    tok_b = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    check_tokenizer_compatibility(tok_a, tok_b)
