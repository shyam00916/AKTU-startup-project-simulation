"""Experiment configuration, deterministic seeding, and environment metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class DataSimulationConfig(BaseModel):
    """Configuration for longitudinal synthetic B2B receivables generation."""

    n_customers: int = 600
    horizon_days: int = 540
    min_invoices_per_customer: int = 10
    max_invoices_per_customer: int = 28
    start_date: str = "2024-01-01"
    late_threshold_days: int = 7
    train_quantile: float = 0.65
    val_quantile: float = 0.80
    min_history_len: int = 3
    max_seq_len: int = 20


class SequenceModelConfig(BaseModel):
    """Hyperparameters for PyTorch Temporal GRU+Attention sequence model."""

    hidden_dim: int = 48
    embedding_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.15
    batch_size: int = 128
    epochs: int = 22
    learning_rate: float = 1.5e-3
    weight_decay: float = 1e-4
    delay_loss_weight: float = 0.08
    early_stopping_patience: int = 5


class TabularModelsConfig(BaseModel):
    """Hyperparameters for Logistic Regression, Random Forest, and XGBoost."""

    logreg_c: float = 0.8
    ridge_alpha: float = 2.0
    rf_n_estimators: int = 220
    rf_max_depth: int = 10
    rf_min_samples_leaf: int = 5
    xgb_n_estimators: int = 240
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.06
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85


class ClusteringConfig(BaseModel):
    """Configuration for K-Means and DBSCAN customer payment behavior clustering."""

    kmeans_k_min: int = 3
    kmeans_k_max: int = 8
    kmeans_target_k: int = 5
    dbscan_eps_grid: list[float] = Field(default_factory=lambda: [1.6, 1.9, 2.2, 2.5, 2.8, 3.2])
    dbscan_min_samples: int = 12


class BanditConfig(BaseModel):
    """Configuration for contextual bandit collection optimization simulation."""

    n_episodes: int = 3500
    linucb_alpha: float = 0.9
    exploration_epsilon: float = 0.15
    goodwill_penalty_weight: float = 1.0


class ExplainabilityConfig(BaseModel):
    """Configuration for SHAP and high-risk customer explainability."""

    high_risk_threshold: float = 0.55
    max_high_risk_customers_export: int = 100
    shap_background_samples: int = 100


class RobustnessConfig(BaseModel):
    """Configuration for distribution shift and robustness stress tests."""

    delay_shift_days: list[float] = Field(default_factory=lambda: [5.0, 10.0, 20.0])
    missing_record_rates: list[float] = Field(default_factory=lambda: [0.20, 0.40, 0.60])
    behavior_drift_fraction: float = 0.35
    industry_shift_high_risk_boost: float = 3.5
    shock_delay_jump_days: float = 14.0
    shock_amount_multiplier: float = 1.45
    shock_partial_pay_boost: float = 0.30


class ExperimentConfig(BaseModel):
    """Master reproducible experiment configuration."""

    experiment_name: str = "receivables_os_ml_v1"
    model_version: str = "1.0.0"
    seed: int = 42
    output_root: str = "artifacts"
    data: DataSimulationConfig = Field(default_factory=DataSimulationConfig)
    sequence: SequenceModelConfig = Field(default_factory=SequenceModelConfig)
    tabular: TabularModelsConfig = Field(default_factory=TabularModelsConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    bandit: BanditConfig = Field(default_factory=BanditConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)

    def config_hash(self) -> str:
        """Compute deterministic SHA-256 short digest of the full configuration."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds across stdlib, NumPy, and PyTorch when available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def capture_environment_metadata() -> dict[str, Any]:
    """Capture runtime Python and ML package versions for reproducibility."""
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "shap",
        "torch",
        "pydantic",
        "streamlit",
        "plotly",
    ]
    versions: dict[str, str] = {}
    import importlib.metadata

    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load ExperimentConfig from JSON/YAML path, or return defaults."""
    if path is None:
        return ExperimentConfig()
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)
    return ExperimentConfig.model_validate(data)
