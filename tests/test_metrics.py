"""Tests for evaluation metrics."""

import pytest

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
    MetricTracker,
)


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("HELLO") == "hello"

    def test_strip_whitespace(self):
        assert normalize_answer("  hello  ") == "hello"

    def test_remove_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("the quick brown fox") == "quick brown fox"
        assert normalize_answer("a cat") == "cat"
        assert normalize_answer("an apple") == "apple"

    def test_collapse_whitespace(self):
        assert normalize_answer("hello    world") == "hello world"

    def test_empty_string(self):
        assert normalize_answer("") == ""


class TestExactMatch:
    def test_identical(self):
        assert exact_match("France", "France") is True

    def test_case_insensitive(self):
        assert exact_match("FRANCE", "france") is True

    def test_with_articles(self):
        assert exact_match("The France", "France") is True

    def test_different(self):
        assert exact_match("France", "Germany") is False

    def test_partial_match_fails(self):
        assert exact_match("Paris, France", "France") is False


class TestExactMatchAny:
    def test_matches_first(self):
        assert exact_match_any("France", ["France", "Germany"]) is True

    def test_matches_second(self):
        assert exact_match_any("Germany", ["France", "Germany"]) is True

    def test_no_match(self):
        assert exact_match_any("Italy", ["France", "Germany"]) is False

    def test_empty_list(self):
        assert exact_match_any("France", []) is False


class TestContainsAnswer:
    def test_exact_contains(self):
        assert contains_answer("The answer is France", "France") is True

    def test_not_contained(self):
        assert contains_answer("The answer is Germany", "France") is False

    def test_empty_ground_truth(self):
        assert contains_answer("anything", "") is True


class TestExtractNumber:
    def test_integer(self):
        assert extract_number("The answer is 42") == 42.0

    def test_decimal(self):
        assert extract_number("The result is 3.14159") == 3.14159

    def test_negative(self):
        assert extract_number("Temperature is -5 degrees") == -5.0

    def test_with_commas(self):
        assert extract_number("Population is 1,234,567") == 1234567.0

    def test_last_number(self):
        assert extract_number("Step 1: 10, Step 2: 20, Final: 30") == 30.0

    def test_no_number(self):
        assert extract_number("no numbers here") is None

    def test_empty_string(self):
        assert extract_number("") is None


class TestNumericalMatch:
    def test_exact_match(self):
        assert numerical_match("42", "42") is True

    def test_with_text(self):
        assert numerical_match("The answer is 42", "42") is True

    def test_different_numbers(self):
        assert numerical_match("42", "43") is False

    def test_floating_point(self):
        assert numerical_match("3.14159", "3.14159") is True

    def test_with_tolerance(self):
        assert numerical_match("3.14", "3.141", tolerance=0.01) is True

    def test_no_number_in_prediction(self):
        assert numerical_match("no number", "42") is False


class TestF1Score:
    def test_identical(self):
        assert f1_score("hello world", "hello world") == 1.0

    def test_partial_overlap(self):
        score = f1_score("the quick brown", "quick brown fox")
        assert 0.5 < score < 1.0

    def test_no_overlap(self):
        assert f1_score("hello", "goodbye") == 0.0

    def test_empty_both(self):
        assert f1_score("", "") == 1.0

    def test_empty_one(self):
        assert f1_score("hello", "") == 0.0
        assert f1_score("", "hello") == 0.0


class TestBatchMetrics:
    def test_batch_exact_match(self):
        preds = ["France", "Germany", "Italy"]
        gts = ["France", "Germany", "Spain"]

        assert batch_exact_match(preds, gts) == pytest.approx(2/3)

    def test_batch_numerical_match(self):
        preds = ["42", "The answer is 100", "50"]
        gts = ["42", "100", "60"]

        assert batch_numerical_match(preds, gts) == pytest.approx(2/3)

    def test_batch_f1_score(self):
        preds = ["hello world", "foo bar"]
        gts = ["hello world", "foo bar"]

        assert batch_f1_score(preds, gts) == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            batch_exact_match(["a", "b"], ["a"])

    def test_empty_list(self):
        assert batch_exact_match([], []) == 0.0


class TestMetricTracker:
    def test_add_and_accuracy(self):
        tracker = MetricTracker()
        tracker.add("France", "France", "test_1")
        tracker.add("Germany", "France", "test_2")

        assert tracker.exact_match_accuracy() == 0.5

    def test_add_batch(self):
        tracker = MetricTracker()
        tracker.add_batch(
            ["France", "Germany"],
            ["France", "Germany"],
        )

        assert tracker.exact_match_accuracy() == 1.0

    def test_get_errors(self):
        tracker = MetricTracker()
        tracker.add("France", "France", "correct")
        tracker.add("Germany", "France", "wrong")

        errors = tracker.get_errors()
        assert len(errors) == 1
        assert errors[0][0] == "wrong"  # task_id

    def test_summary(self):
        tracker = MetricTracker()
        tracker.add("France", "France")
        tracker.add("42", "42")

        summary = tracker.summary()
        assert summary["n_samples"] == 2
        assert summary["exact_match"] == 1.0

    def test_reset(self):
        tracker = MetricTracker()
        tracker.add("France", "France")
        tracker.reset()

        assert len(tracker.predictions) == 0
