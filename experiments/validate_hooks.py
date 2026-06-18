#!/usr/bin/env python3
# pylint: disable=redefined-outer-name
"""
Validate hook mechanism for activation extraction and injection.

This script tests that:
1. Activation extraction works correctly
2. Activation injection modifies the output
3. The full grafting pipeline is functional

Run this before any experiments to ensure the infrastructure works.

Usage:
    python experiments/validate_hooks.py
    python experiments/validate_hooks.py --model Qwen/Qwen2.5-0.5B
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.activation_extractor import (  # pylint: disable=wrong-import-position
    ActivationExtractor,
    ActivationInjector,
    ExtractionPoint,
)
from src.models.model_loader import load_model_and_tokenizer, get_model_info  # pylint: disable=wrong-import-position
from src.utils.config import ModelConfig, set_seed  # pylint: disable=wrong-import-position


def parse_args():
    parser = argparse.ArgumentParser(description="Validate hook mechanism")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Model to test")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use")
    return parser.parse_args()


def test_activation_extraction(model, tokenizer, device):
    """Test that activation extraction works correctly."""
    print("\n" + "=" * 50)
    print("Test 1: Activation Extraction")
    print("=" * 50)

    text = "The capital of France is"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    n_layers = model.config.num_hidden_layers
    test_layers = [0, n_layers // 2, n_layers - 1]

    print(f"Input: '{text}'")
    print(f"Testing layers: {test_layers}")

    # Test POST_MLP extraction (most common)
    extractor = ActivationExtractor(
        model, test_layers,
        extraction_point=ExtractionPoint.POST_MLP,
        token_position=-1,
    )

    activations = extractor.extract(inputs["input_ids"], inputs.get("attention_mask"))

    print("\nExtracted activations:")
    for layer_idx, act in activations.items():
        print(f"  Layer {layer_idx}: shape={act.tensor.shape}, norm={act.tensor.norm().item():.4f}")

        # Verify shape
        expected_d = model.config.hidden_size
        assert act.tensor.shape == (1, expected_d), f"Wrong shape: {act.tensor.shape}"
        assert act.tensor.norm() > 0, f"Zero activation at layer {layer_idx}"

    print("\n✓ Activation extraction works correctly!")
    return True


def test_extraction_points(model, tokenizer, device):
    """Test all extraction points."""
    print("\n" + "=" * 50)
    print("Test 2: Extraction Points")
    print("=" * 50)

    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    layer = model.config.num_hidden_layers // 2

    for point in ExtractionPoint:
        try:
            extractor = ActivationExtractor(
                model, [layer],
                extraction_point=point,
                token_position=-1,
            )

            acts = extractor.extract(inputs["input_ids"])
            norm = acts[layer].tensor.norm().item()
            print(f"  {point.value}: norm={norm:.4f} ✓")

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"  {point.value}: FAILED - {e}")
            return False

    print("\n✓ All extraction points work!")
    return True


def test_activation_injection(model, tokenizer, device):
    """Test that activation injection modifies output."""
    print("\n" + "=" * 50)
    print("Test 3: Activation Injection")
    print("=" * 50)

    text = "The answer is"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    layer = model.config.num_hidden_layers // 2
    d_model = model.config.hidden_size

    # Get baseline output
    with torch.no_grad():
        baseline_output = model(**inputs)
        baseline_logits = baseline_output.logits[0, -1, :]

    # Create random injection
    random_activation = torch.randn(1, d_model, device=device)

    # Inject and get new output
    injector = ActivationInjector(model, layer, ExtractionPoint.POST_MLP)

    with injector:
        injector.set_injection(random_activation, position=-1)
        with torch.no_grad():
            injected_output = model(**inputs)
            injected_logits = injected_output.logits[0, -1, :]

    # Verify outputs differ
    cosine_sim = F.cosine_similarity(
        baseline_logits.unsqueeze(0),
        injected_logits.unsqueeze(0),
    ).item()

    print(f"  Baseline logits norm: {baseline_logits.norm().item():.4f}")
    print(f"  Injected logits norm: {injected_logits.norm().item():.4f}")
    print(f"  Cosine similarity: {cosine_sim:.4f}")

    # They should be different (cosine < 0.99)
    if cosine_sim < 0.99:
        print("\n✓ Activation injection modifies output!")
        return True

    print("\n✗ Injection didn't change output!")
    return False


def test_multiple_forward_passes(model, tokenizer, device):
    """Test that extraction works across multiple forward passes."""
    print("\n" + "=" * 50)
    print("Test 4: Multiple Forward Passes")
    print("=" * 50)

    texts = ["Hello", "The quick brown fox", "What is AI?"]
    layer = model.config.num_hidden_layers // 2

    extractor = ActivationExtractor(
        model, [layer],
        extraction_point=ExtractionPoint.POST_MLP,
    )

    activations = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        acts = extractor.extract(inputs["input_ids"])
        activations.append(acts[layer].tensor)

        print(f"  '{text}': norm={acts[layer].tensor.norm().item():.4f}")

    # Check they're all different
    for i, act_i in enumerate(activations):
        for j in range(i + 1, len(activations)):
            sim = F.cosine_similarity(
                act_i, activations[j]
            ).item()
            if sim > 0.99:
                print(f"\n✗ Activations for '{texts[i]}' and '{texts[j]}' are too similar!")
                return False

    print("\n✓ Multiple forward passes work correctly!")
    return True


def test_generation_with_hooks(model, tokenizer, device):
    """Test that generation works with hooks registered."""
    print("\n" + "=" * 50)
    print("Test 5: Generation with Hooks")
    print("=" * 50)

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    layer = model.config.num_hidden_layers // 2

    # Create extractor
    extractor = ActivationExtractor(model, [layer])

    # Generate without hooks (baseline)
    with torch.no_grad():
        baseline_output = model.generate(
            **inputs,
            max_new_tokens=10,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)

    # Generate with hooks registered
    with extractor:
        with torch.no_grad():
            hooked_output = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    hooked_text = tokenizer.decode(hooked_output[0], skip_special_tokens=True)

    print(f"  Baseline: {baseline_text}")
    print(f"  With hooks: {hooked_text}")

    # They should be identical (extraction shouldn't change output)
    if baseline_text == hooked_text:
        print("\n✓ Generation works correctly with hooks!")
        return True

    print("\n✗ Hooks changed generation output!")
    return False


def main():
    args = parse_args()
    set_seed(42)

    print("=" * 60)
    print("Hook Validation Tests")
    print("=" * 60)
    print(f"Model: {args.model}")

    # Load model
    print("\nLoading model...")
    config = ModelConfig(name=args.model)
    model, tokenizer = load_model_and_tokenizer(config, args.device)
    device = next(model.parameters()).device

    info = get_model_info(model)
    print(f"  Layers: {info['n_layers']}")
    print(f"  Hidden size: {info['d_model']}")
    print(f"  Device: {device}")

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run tests
    tests = [
        ("Activation Extraction", lambda: test_activation_extraction(model, tokenizer, device)),
        ("Extraction Points", lambda: test_extraction_points(model, tokenizer, device)),
        ("Activation Injection", lambda: test_activation_injection(model, tokenizer, device)),
        ("Multiple Forward Passes", lambda: test_multiple_forward_passes(model, tokenizer, device)),
        ("Generation with Hooks", lambda: test_generation_with_hooks(model, tokenizer, device)),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"\n✗ {name} failed with exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed! Infrastructure is ready.")
        return 0

    print("Some tests failed. Please fix issues before running experiments.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
