"""Tests for combination functions."""

import pytest
import torch

from src.communication.combination_functions import (
    SumCombination,
    MeanCombination,
    ReplaceCombination,
    WeightedSumCombination,
    LearnedLinearCombination,
    ConcatProjectCombination,
    GatedCombination,
    get_combination_function,
)


class TestSumCombination:
    def test_same_dimensions(self):
        fn = SumCombination()
        a = torch.randn(2, 768)
        b = torch.randn(2, 768)

        result = fn.combine(a, b)

        assert result.shape == (2, 768)
        assert torch.allclose(result, a + b)

    def test_dimension_mismatch_raises(self):
        fn = SumCombination()
        a = torch.randn(2, 512)
        b = torch.randn(2, 768)

        with pytest.raises(ValueError):
            fn.combine(a, b)

    def test_not_trainable(self):
        fn = SumCombination()
        assert fn.trainable is False
        assert not list(fn.parameters())


class TestMeanCombination:
    def test_same_dimensions(self):
        fn = MeanCombination()
        a = torch.randn(2, 768)
        b = torch.randn(2, 768)

        result = fn.combine(a, b)

        assert result.shape == (2, 768)
        assert torch.allclose(result, 0.5 * (a + b))

    def test_dimension_mismatch_raises(self):
        fn = MeanCombination()
        a = torch.randn(2, 512)
        b = torch.randn(2, 768)

        with pytest.raises(ValueError):
            fn.combine(a, b)


class TestReplaceCombination:
    def test_same_dimensions(self):
        fn = ReplaceCombination()
        a = torch.randn(2, 768)
        b = torch.randn(2, 768)

        result = fn.combine(a, b)

        assert result.shape == (2, 768)
        assert torch.allclose(result, a)

    def test_different_dimensions_projects(self):
        fn = ReplaceCombination(d_a=512, d_b=768)
        a = torch.randn(2, 512)
        b = torch.randn(2, 768)

        result = fn.combine(a, b)

        assert result.shape == (2, 768)

    def test_lazy_initialization(self):
        fn = ReplaceCombination()  # No dimensions specified
        a = torch.randn(2, 512)
        b = torch.randn(2, 768)

        result = fn.combine(a, b)

        assert result.shape == (2, 768)


class TestWeightedSumCombination:
    def test_initial_weights(self):
        fn = WeightedSumCombination(init_alpha=0.0)

        # sigmoid(0) = 0.5, so should be equal weighting
        assert abs(fn.alpha - 0.5) < 0.01

    def test_combine(self):
        fn = WeightedSumCombination(init_alpha=0.0)
        a = torch.ones(2, 768)
        b = torch.zeros(2, 768)

        result = fn.combine(a, b)

        # Should be approximately 0.5 * 1 + 0.5 * 0 = 0.5
        assert torch.allclose(result, torch.full_like(result, 0.5), atol=0.01)

    def test_trainable(self):
        fn = WeightedSumCombination()
        assert fn.trainable is True
        params = list(fn.parameters())
        assert len(params) == 1


class TestLearnedLinearCombination:
    def test_output_shape(self):
        fn = LearnedLinearCombination(d_a=512, d_b=768)
        a = torch.randn(2, 512)
        b = torch.randn(2, 768)

        result = fn.combine(a, b)

        assert result.shape == (2, 768)

    def test_trainable(self):
        fn = LearnedLinearCombination(d_a=512, d_b=768)
        assert fn.trainable is True
        params = list(fn.parameters())
        assert len(params) > 0

    def test_state_dict(self):
        fn = LearnedLinearCombination(d_a=512, d_b=768)
        state = fn.state_dict()

        assert "projection" in state

        # Load back
        fn2 = LearnedLinearCombination(d_a=512, d_b=768)
        fn2.load_state_dict(state)

        # Weights should match
        assert torch.allclose(
            fn.projection.weight,
            fn2.projection.weight,
        )


class TestConcatProjectCombination:
    def test_output_shape(self):
        fn = ConcatProjectCombination(d_a=512, d_b=768)
        a = torch.randn(2, 512)
        b = torch.randn(2, 768)

        result = fn.combine(a, b)

        assert result.shape == (2, 768)

    def test_trainable(self):
        fn = ConcatProjectCombination(d_a=512, d_b=768)
        assert fn.trainable is True
        params = list(fn.parameters())
        assert len(params) > 0


class TestGatedCombination:
    def test_output_shape(self):
        fn = GatedCombination(d_a=512, d_b=768)
        a = torch.randn(2, 512)
        b = torch.randn(2, 768)

        result = fn.combine(a, b)

        assert result.shape == (2, 768)

    def test_initial_bias_favors_b(self):
        """Gate should initially favor B (bias is negative)."""
        fn = GatedCombination(d_a=512, d_b=768)

        # With zeros input, gate should be sigmoid(-2) ≈ 0.12
        a = torch.zeros(1, 512)
        b = torch.ones(1, 768)

        result = fn.combine(a, b)

        # Result should be closer to b than to a
        # Since gate ≈ 0.12, result ≈ 0.12 * proj(a) + 0.88 * b ≈ 0.88
        assert result.mean() > 0.5


class TestGetCombinationFunction:
    def test_sum(self):
        fn = get_combination_function("sum", d_a=768, d_b=768)
        assert isinstance(fn, SumCombination)

    def test_mean(self):
        fn = get_combination_function("mean", d_a=768, d_b=768)
        assert isinstance(fn, MeanCombination)

    def test_replace(self):
        fn = get_combination_function("replace", d_a=512, d_b=768)
        assert isinstance(fn, ReplaceCombination)

    def test_learned_linear(self):
        fn = get_combination_function("learned_linear", d_a=512, d_b=768)
        assert isinstance(fn, LearnedLinearCombination)

    def test_concat_project(self):
        fn = get_combination_function("concat_project", d_a=512, d_b=768)
        assert isinstance(fn, ConcatProjectCombination)

    def test_gated(self):
        fn = get_combination_function("gated", d_a=512, d_b=768)
        assert isinstance(fn, GatedCombination)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_combination_function("unknown", d_a=768, d_b=768)

    def test_dimension_mismatch_for_sum(self):
        with pytest.raises(ValueError):
            get_combination_function("sum", d_a=512, d_b=768)


class TestDeviceMovement:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device(self):
        fn = LearnedLinearCombination(d_a=512, d_b=768)
        fn.to(torch.device("cuda"))

        a = torch.randn(2, 512, device="cuda")
        b = torch.randn(2, 768, device="cuda")

        result = fn.combine(a, b)

        assert result.device.type == "cuda"
