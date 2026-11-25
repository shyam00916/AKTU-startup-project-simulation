"""Evaluation metrics, model comparison, and robustness testing package."""

from .comparison import MODEL_DISPLAY_NAMES, ComparisonArtifacts, run_model_comparison
from .metrics import (
    ModelPrediction,
    compute_calibration_metrics,
    evaluate_predictions,
    find_optimal_f1_threshold,
)
from .robustness import run_robustness_suite

__all__ = [
    "MODEL_DISPLAY_NAMES",
    "ComparisonArtifacts",
    "ModelPrediction",
    "compute_calibration_metrics",
    "evaluate_predictions",
    "find_optimal_f1_threshold",
    "run_model_comparison",
    "run_robustness_suite",
]
