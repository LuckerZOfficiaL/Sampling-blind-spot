"""Communication modules for activation grafting."""

from src.communication.combination_functions import (
    CombinationFunction,
    SumCombination,
    MeanCombination,
    ReplaceCombination,
    TrainedProjectionCombination,
    WeightedSumCombination,
    LearnedLinearCombination,
    ConcatProjectCombination,
    GatedCombination,
    get_combination_function,
)
from src.communication.activation_graft import (
    GraftingMode,
    GraftingDiagnostics,
    ActivationGraftingEngine,
    create_grafting_engine,
)

__all__ = [
    "CombinationFunction",
    "SumCombination",
    "MeanCombination",
    "ReplaceCombination",
    "TrainedProjectionCombination",
    "WeightedSumCombination",
    "LearnedLinearCombination",
    "ConcatProjectCombination",
    "GatedCombination",
    "get_combination_function",
    "GraftingMode",
    "GraftingDiagnostics",
    "ActivationGraftingEngine",
    "create_grafting_engine",
]
