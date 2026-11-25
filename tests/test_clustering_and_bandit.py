"""Tests for customer behavior clustering (K-Means, DBSCAN) and contextual bandit simulation."""

from __future__ import annotations

from receivables_ml.bandit import ACTIONS, EPISTEMIC_DISCLAIMER, run_collection_bandit_simulation
from receivables_ml.config import BanditConfig, ClusteringConfig, DataSimulationConfig, TabularModelsConfig
from receivables_ml.data import generate_receivables_dataset, prepare_datasets
from receivables_ml.models import fit_clustering_and_segmentation


def test_clustering_personas_and_dbscan() -> None:
    d_cfg = DataSimulationConfig(n_customers=100, min_invoices_per_customer=10, max_invoices_per_customer=14)
    _, tx = generate_receivables_dataset(d_cfg, seed=42)
    ds = prepare_datasets(tx, d_cfg)

    cl_cfg = ClusteringConfig(kmeans_k_min=3, kmeans_k_max=6, kmeans_target_k=5, dbscan_min_samples=6)
    tab_cfg = TabularModelsConfig(rf_n_estimators=20, xgb_n_estimators=30)
    cl = fit_clustering_and_segmentation(ds, cl_cfg, tab_cfg, seed=42)

    assert len(cl.cluster_profiles) == 5
    personas = {p["segment_name"] for p in cl.cluster_profiles}
    assert personas == {
        "Consistent Early Payers",
        "Consistent Late Payers",
        "Irregular Payers",
        "High-Value Delayed Payers",
        "Deteriorating Customers",
    }
    assert "n_outliers" in cl.dbscan_summary
    assert "xgboost" in cl.segmentation_lift["model_feature_lift"]


def test_contextual_bandit_ope_and_disclaimer() -> None:
    d_cfg = DataSimulationConfig(n_customers=80, min_invoices_per_customer=10, max_invoices_per_customer=14)
    _, tx = generate_receivables_dataset(d_cfg, seed=42)
    ds = prepare_datasets(tx, d_cfg)

    report = run_collection_bandit_simulation(
        customer_pool_meta=ds.test.meta,
        config=BanditConfig(n_episodes=600),
        seed=42,
    )
    assert report.disclaimer == EPISTEMIC_DISCLAIMER
    assert "DO NOT prove real-world causal effectiveness" in report.disclaimer
    assert set(report.action_costs.keys()) == set(ACTIONS)
    policy_names = {r["policy"] for r in report.policy_comparison}
    assert {
        "Logging Policy (Historical Heuristic)",
        "Direct-Method Outcome Policy",
        "LinUCB Contextual Bandit",
        "Doubly-Robust Policy Optimizer",
    }.issubset(policy_names)
    for row in report.policy_comparison:
        assert 0.0 <= row["estimated_payment_prob_model"] <= 1.0
        assert 0.0 <= row["sim_oracle_payment_prob"] <= 1.0
