"""End-to-end integration test for SHAP explainability, robustness suite, and reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

from receivables_ml.config import (
    BanditConfig,
    ClusteringConfig,
    DataSimulationConfig,
    ExperimentConfig,
    ExplainabilityConfig,
    RobustnessConfig,
    SequenceModelConfig,
    TabularModelsConfig,
)
from receivables_ml.pipeline import run_full_experiment


def test_end_to_end_reproducibility_shap_and_robustness(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        experiment_name="test_integration",
        model_version="0.1.0-test",
        seed=123,
        output_root="artifacts",
        data=DataSimulationConfig(n_customers=85, min_invoices_per_customer=10, max_invoices_per_customer=14),
        sequence=SequenceModelConfig(hidden_dim=24, embedding_dim=16, epochs=4, batch_size=64),
        tabular=TabularModelsConfig(rf_n_estimators=25, xgb_n_estimators=35),
        clustering=ClusteringConfig(kmeans_k_min=3, kmeans_k_max=5, kmeans_target_k=5, dbscan_min_samples=6),
        bandit=BanditConfig(n_episodes=500),
        explainability=ExplainabilityConfig(high_risk_threshold=0.50, max_high_risk_customers_export=20, shap_background_samples=40),
        robustness=RobustnessConfig(delay_shift_days=[5.0, 15.0], missing_record_rates=[0.25, 0.50]),
    )

    out = run_full_experiment(cfg, base_dir=tmp_path)
    run_dir = Path(out["run_dir"])

    assert (run_dir / "config.json").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "reports" / "high_risk_explanations.json").exists()
    assert (run_dir / "reports" / "robustness_report.md").exists()
    assert (run_dir / "reports" / "experiment_summary.md").exists()
    assert (tmp_path / "artifacts" / "registry.json").exists()

    hr_data = json.loads((run_dir / "reports" / "high_risk_explanations.json").read_text(encoding="utf-8"))
    assert len(hr_data) > 0
    first_hr = hr_data[0]
    for required_field in (
        "most_influential_risk_factors",
        "historical_payment_trends",
        "outstanding_balance_contribution",
        "previous_payment_delays_contribution",
    ):
        assert required_field in first_hr

    metrics = out["metrics"]
    assert set(metrics["model_comparison"]["test_metrics"].keys()) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
        "sequence_model",
    }
    rob_categories = {r["category"] for r in metrics["robustness"]["stress_test_results"]}
    assert rob_categories == {
        "Increased Payment Delays",
        "Industry Distribution Change",
        "Customer Behavior Drift",
        "Missing Transaction Records",
        "Sudden Economic Shock",
    }
