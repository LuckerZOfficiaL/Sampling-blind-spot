#!/usr/bin/env python3
# pylint: disable=redefined-outer-name
"""
Layer grid search for finding optimal (layer_a, layer_b) pairs.

This script searches over layer combinations to find which layers
produce the best activation grafting performance.

Usage:
    python experiments/layer_grid_search.py --model Qwen/Qwen2.5-0.5B
    python experiments/layer_grid_search.py --model Qwen/Qwen2.5-0.5B --stride 4  # Coarse search
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.reasoning_tasks import create_gsm8k_prompts, load_gsm8k_subset  # pylint: disable=wrong-import-position
from src.communication.activation_graft import ActivationGraftingEngine, GraftingMode  # pylint: disable=wrong-import-position
from src.communication.combination_functions import get_combination_function  # pylint: disable=wrong-import-position
from src.evaluation.metrics import numerical_match  # pylint: disable=wrong-import-position
from src.models.activation_extractor import ExtractionPoint  # pylint: disable=wrong-import-position
from src.models.model_loader import load_model_and_tokenizer  # pylint: disable=wrong-import-position
from src.models.model_registry import get_model_spec  # pylint: disable=wrong-import-position
from src.utils.config import ModelConfig, set_seed  # pylint: disable=wrong-import-position
from src.utils.wandb_utils import WandbRun, init_wandb, timer  # pylint: disable=wrong-import-position


def parse_args():
    parser = argparse.ArgumentParser(description="Layer grid search")

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Model to use (same for A and B)")
    parser.add_argument("--model-a", type=str, default=None,
                        help="Model A (overrides --model)")
    parser.add_argument("--model-b", type=str, default=None,
                        help="Model B (overrides --model)")

    # Search configuration
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride for layer search (>1 for coarse search)")
    parser.add_argument("--layers-a", type=str, default=None,
                        help="Specific layers for A (comma-separated, e.g., '4,8,12')")
    parser.add_argument("--layers-b", type=str, default=None,
                        help="Specific layers for B (comma-separated)")

    # Benchmark
    parser.add_argument("--n-samples", type=int, default=30,
                        help="Samples per layer pair (smaller for faster search)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Combination function
    parser.add_argument("--combination-fn", type=str, default="sum",
                        help="Combination function")

    # Output
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--save-heatmap", action="store_true",
                        help="Save heatmap visualization")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging")

    return parser.parse_args()


def evaluate_layer_pair(
    model_a,
    model_b,
    tokenizer_a,
    tokenizer_b,
    layer_a: int,
    layer_b: int,
    combination_fn,
    examples,
    max_new_tokens: int = 50,
) -> float:
    """Evaluate accuracy for a specific layer pair."""
    # Create engine
    engine = ActivationGraftingEngine(
        model_a=model_a,
        model_b=model_b,
        tokenizer_a=tokenizer_a,
        tokenizer_b=tokenizer_b,
        layer_a=layer_a,
        layer_b=layer_b,
        combination_fn=combination_fn,
        extraction_point=ExtractionPoint.POST_MLP,
        grafting_mode=GraftingMode.SINGLE_SHOT,
    )

    correct = 0
    for ex in examples:
        prompt_a, prompt_b = create_gsm8k_prompts(ex)
        output = engine.generate(
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            max_new_tokens=max_new_tokens,
        )
        if numerical_match(output, str(ex.numerical_answer)):
            correct += 1

    return correct / len(examples)


def run_grid_search(
    model_a,
    model_b,
    tokenizer_a,
    tokenizer_b,
    layers_a: List[int],
    layers_b: List[int],
    d_a: int,
    d_b: int,
    combination_fn_name: str,
    examples,
    wb_run: WandbRun = None,
) -> Tuple[Dict[Tuple[int, int], float], Tuple[int, int], float]:
    """
    Run grid search over layer pairs.

    Returns:
        Tuple of (results_dict, best_pair, best_accuracy)
    """
    if wb_run is None:
        wb_run = WandbRun()
    results = {}
    best_pair = None
    best_accuracy = -1

    total = len(layers_a) * len(layers_b)
    current = 0

    for la in layers_a:
        for lb in layers_b:
            current += 1
            print(f"  [{current}/{total}] Layer pair ({la}, {lb})...", end=" ", flush=True)

            # Create combination function
            combination_fn = get_combination_function(
                combination_fn_name, d_a, d_b
            )

            accuracy = evaluate_layer_pair(
                model_a, model_b,
                tokenizer_a, tokenizer_b,
                la, lb,
                combination_fn,
                examples,
            )

            results[(la, lb)] = accuracy
            print(f"accuracy = {accuracy:.2%}")

            # Log per-pair accuracy
            wb_run.log({
                "grid/layer_a": la,
                "grid/layer_b": lb,
                "grid/accuracy": accuracy,
            }, step=current)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_pair = (la, lb)

    return results, best_pair, best_accuracy


def create_heatmap(
    results: Dict[Tuple[int, int], float],
    layers_a: List[int],
    layers_b: List[int],
    output_path: str,
):
    """Create and save heatmap visualization."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        import seaborn as sns  # pylint: disable=import-outside-toplevel
    except ImportError:
        print("matplotlib/seaborn not available, skipping heatmap")
        return

    # Create matrix
    matrix = np.zeros((len(layers_a), len(layers_b)))
    for i, la in enumerate(layers_a):
        for j, lb in enumerate(layers_b):
            matrix[i, j] = results.get((la, lb), 0)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        xticklabels=layers_b,
        yticklabels=layers_a,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Layer B (receiver)")
    plt.ylabel("Layer A (sender)")
    plt.title("Accuracy by Layer Pair")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to: {output_path}")


def main():  # pylint: disable=too-many-statements
    args = parse_args()
    set_seed(args.seed)

    # Determine models
    model_a_name = args.model_a or args.model
    model_b_name = args.model_b or args.model

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"layer_search_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    wb_run = init_wandb(
        enabled=args.wandb,
        project="llm-activations",
        name=f"layer-grid-{model_a_name.split('/')[-1]}",
        config={
            "experiment_type": "layer_grid_search",
            "model_a": model_a_name,
            "model_b": model_b_name,
            "stride": args.stride,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "combination_fn": args.combination_fn,
        },
        tags=["layer-grid-search", "gsm8k"],
    )

    print("=" * 60)
    print("Layer Grid Search")
    print("=" * 60)
    print(f"Model A: {model_a_name}")
    print(f"Model B: {model_b_name}")
    print(f"Stride: {args.stride}")
    print("Benchmark: GSM8K")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {output_dir}")
    print()

    # Get model specs
    spec_a = get_model_spec(model_a_name)
    spec_b = get_model_spec(model_b_name)

    # Determine layers to search
    if args.layers_a:
        layers_a = [int(x) for x in args.layers_a.split(",")]
    else:
        layers_a = list(range(0, spec_a.n_layers, args.stride))

    if args.layers_b:
        layers_b = [int(x) for x in args.layers_b.split(",")]
    else:
        layers_b = list(range(0, spec_b.n_layers, args.stride))

    print(f"Searching layers A: {layers_a}")
    print(f"Searching layers B: {layers_b}")
    print(f"Total combinations: {len(layers_a) * len(layers_b)}")
    print()

    # Load models
    print("Loading models...")
    with timer(wb_run, "time/model_load_s"):
        model_a, tokenizer_a = load_model_and_tokenizer(
            ModelConfig(name=model_a_name)
        )
        if model_a_name == model_b_name:
            model_b, tokenizer_b = model_a, tokenizer_a
        else:
            model_b, tokenizer_b = load_model_and_tokenizer(
                ModelConfig(name=model_b_name)
            )
    print()

    # Load benchmark
    print("Loading GSM8K benchmark...")
    examples = load_gsm8k_subset(n_samples=args.n_samples, seed=args.seed)
    print(f"Loaded {len(examples)} examples")
    print()

    # Run grid search
    print("Running grid search...")
    with timer(wb_run, "time/grid_search_s"):
        results, best_pair, best_accuracy = run_grid_search(
            model_a, model_b,
            tokenizer_a, tokenizer_b,
            layers_a, layers_b,
            spec_a.d_model, spec_b.d_model,
            args.combination_fn,
            examples,
            wb_run=wb_run,
        )

    # Save results
    results_serializable = {
        f"({la},{lb})": acc for (la, lb), acc in results.items()
    }
    results_data = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "layers_a": layers_a,
        "layers_b": layers_b,
        "combination_fn": args.combination_fn,
        "n_samples": args.n_samples,
        "benchmark": "gsm8k",
        "results": results_serializable,
        "best_pair": list(best_pair),
        "best_accuracy": best_accuracy,
    }

    with open(output_dir / "results.json", 'w', encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    # Log grid search results as W&B table
    wb_run.log_table(
        key="grid_search_results",
        columns=["layer_a", "layer_b", "accuracy"],
        data=[
            [la, lb, acc] for (la, lb), acc in results.items()
        ],
    )

    # Log summary
    wb_run.summary_update({
        "best_layer_a": best_pair[0],
        "best_layer_b": best_pair[1],
        "best_accuracy": best_accuracy,
        "total_combinations": len(results),
    })

    # Create heatmap
    if args.save_heatmap:
        heatmap_path = str(output_dir / "heatmap.png")
        create_heatmap(results, layers_a, layers_b, heatmap_path)
        wb_run.log_artifact(
            path=heatmap_path,
            name="layer-grid-heatmap",
            artifact_type="visualization",
        )

    # Log results directory as artifact
    wb_run.log_artifact(
        path=str(output_dir),
        name="layer-grid-results",
        artifact_type="results",
        metadata={"best_accuracy": best_accuracy},
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nBest layer pair: ({best_pair[0]}, {best_pair[1]})")
    print(f"Best accuracy: {best_accuracy:.2%}")

    # Top 5 pairs
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 layer pairs:")
    for (la, lb), acc in sorted_results[:5]:
        print(f"  ({la:2d}, {lb:2d}): {acc:.2%}")

    print(f"\nResults saved to: {output_dir}")
    wb_run.finish()


if __name__ == "__main__":
    main()
