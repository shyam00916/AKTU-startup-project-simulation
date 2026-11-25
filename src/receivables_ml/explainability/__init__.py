"""SHAP explainability and temporal attribution package."""

from .shap_explainer import BALANCE_FEATURES, DELAY_HISTORY_FEATURES, generate_shap_explanations

__all__ = [
    "BALANCE_FEATURES",
    "DELAY_HISTORY_FEATURES",
    "generate_shap_explanations",
]
