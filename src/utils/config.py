"""Configuration dataclasses and utilities for experiments."""

import random
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Any, Dict
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Seed all sources of randomness for reproducibility.

    Call this before any model inference or training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    revision: Optional[str] = None
    torch_dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraftingConfig:
    """Configuration for activation grafting."""

    layer_a: int
    layer_b: int
    extraction_point: str = "POST_MLP"
    grafting_mode: str = "SINGLE_SHOT"
    combination_fn: str = "sum"

    # For learned combination functions
    combination_fn_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlignmentTrainingConfig:
    """Configuration for training learned projections."""

    num_prompts: int = 1000
    num_steps: int = 1000
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    objective: Literal["mse", "cosine", "mse+cosine"] = "mse"
    cosine_weight: float = 0.5
    validation_split: float = 0.1
    early_stopping_patience: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    benchmark: str = "countries"
    n_samples: int = 100
    seed: int = 42
    max_new_tokens: int = 50
    temperature: float = 0.0
    do_sample: bool = False

    # Statistical
    num_seeds: int = 3
    bootstrap_samples: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    # Experiment metadata
    name: str = "experiment"
    description: str = ""

    # Models
    model_a: ModelConfig = field(default_factory=lambda: ModelConfig(name="Qwen/Qwen2.5-0.5B"))
    model_b: ModelConfig = field(default_factory=lambda: ModelConfig(name="Qwen/Qwen2.5-0.5B"))

    # Grafting
    grafting: GraftingConfig = field(default_factory=lambda: GraftingConfig(layer_a=12, layer_b=12))

    # Alignment training (optional)
    alignment: Optional[AlignmentTrainingConfig] = None

    # Evaluation
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Output
    output_dir: str = "results"
    save_predictions: bool = True
    save_diagnostics: bool = True

    # Logging
    use_wandb: bool = False
    wandb_project: str = "llm-activations"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Convert nested dicts to dataclasses
        if 'model_a' in data:
            data['model_a'] = ModelConfig(**data['model_a'])
        if 'model_b' in data:
            data['model_b'] = ModelConfig(**data['model_b'])
        if 'grafting' in data:
            data['grafting'] = GraftingConfig(**data['grafting'])
        if 'alignment' in data and data['alignment'] is not None:
            data['alignment'] = AlignmentTrainingConfig(**data['alignment'])
        if 'evaluation' in data:
            data['evaluation'] = EvaluationConfig(**data['evaluation'])

        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        data = self.to_dict()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'model_a': self.model_a.to_dict(),
            'model_b': self.model_b.to_dict(),
            'grafting': self.grafting.to_dict(),
            'alignment': self.alignment.to_dict() if self.alignment else None,
            'evaluation': self.evaluation.to_dict(),
            'output_dir': self.output_dir,
            'save_predictions': self.save_predictions,
            'save_diagnostics': self.save_diagnostics,
            'use_wandb': self.use_wandb,
            'wandb_project': self.wandb_project,
        }


def create_default_configs() -> None:
    """Create default configuration files."""
    configs_dir = Path("experiments/configs")
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Same model config
    same_model = ExperimentConfig(
        name="same_model",
        description="Both models are identical (Qwen2.5-0.5B)",
        model_a=ModelConfig(name="Qwen/Qwen2.5-0.5B"),
        model_b=ModelConfig(name="Qwen/Qwen2.5-0.5B"),
        grafting=GraftingConfig(layer_a=12, layer_b=12, combination_fn="sum"),
    )
    same_model.to_yaml(str(configs_dir / "same_model.yaml"))

    # Same family config
    same_family = ExperimentConfig(
        name="same_family",
        description="Same family, different sizes (requires projection)",
        model_a=ModelConfig(name="Qwen/Qwen2.5-0.5B"),
        model_b=ModelConfig(name="Qwen/Qwen2.5-1.5B"),
        grafting=GraftingConfig(layer_a=12, layer_b=14, combination_fn="learned_linear"),
        alignment=AlignmentTrainingConfig(num_prompts=1000, num_steps=1000),
    )
    same_family.to_yaml(str(configs_dir / "same_family.yaml"))

    # Cross family config
    cross_family = ExperimentConfig(
        name="cross_family",
        description="Different families (Qwen -> TinyLlama)",
        model_a=ModelConfig(name="Qwen/Qwen2.5-0.5B"),
        model_b=ModelConfig(name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        grafting=GraftingConfig(layer_a=12, layer_b=11, combination_fn="learned_linear"),
        alignment=AlignmentTrainingConfig(num_prompts=1000, num_steps=2000),
    )
    cross_family.to_yaml(str(configs_dir / "cross_family.yaml"))


if __name__ == "__main__":
    create_default_configs()
    print("Created default configuration files in experiments/configs/")
