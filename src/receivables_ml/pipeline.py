"""End-to-end reproducible ReceivablesOS ML research pipeline orchestrator."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from receivables_ml.bandit import run_collection_bandit_simulation
from receivables_ml.config import ExperimentConfig, set_global_seed
from receivables_ml.data import generate_receivables_dataset, prepare_datasets
from receivables_ml.evaluation import (
    MODEL_DISPLAY_NAMES,
    run_model_comparison,
    run_robustness_suite,
)
from receivables_ml.explainability import generate_shap_explanations
from receivables_ml.models import fit_clustering_and_segmentation
from receivables_ml.tracking import ExperimentTracker


def _build_experiment_summary_markdown(
    run_id: str,
    config: ExperimentConfig,
    dataset_summary: dict[str, Any],
    comparison_payload: dict[str, Any],
    clustering_payload: dict[str, Any],
    bandit_payload: dict[str, Any],
    shap_summary: dict[str, Any],
    robustness_payload: dict[str, Any],
) -> str:
    lines = [
        f"# ReceivablesOS ML Intelligence Research Summary (`{run_id}`)",
        "",
        f"- **Model Version**: `{config.model_version}` | **Config Hash**: `{config.config_hash()}` | **Seed**: `{config.seed}`",
        f"- **Customers**: `{dataset_summary['n_customers']}` | **Total Transactions**: `{dataset_summary['n_total_transactions']}` | **Splits (Train/Val/Test)**: `{dataset_summary['n_train']}/{dataset_summary['n_val']}/{dataset_summary['n_test']}`",
        "",
        "## 1. Model Comparison (Holdout Test Set)",
        "",
        "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | ECE (Calib) | Brier | MAE (days) | RMSE (days) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m_key, m_vals in comparison_payload["test_metrics"].items():
        lines.append(
            f"| {MODEL_DISPLAY_NAMES[m_key]} | {m_vals['roc_auc']:.4f} | {m_vals['pr_auc']:.4f} | "
            f"{m_vals['precision']:.4f} | {m_vals['recall']:.4f} | {m_vals['f1']:.4f} | "
            f"{m_vals['calibration_ece']:.4f} | {m_vals['brier_score']:.4f} | "
            f"{m_vals['mae']:.2f} | {m_vals['rmse']:.2f} |"
        )

    emb = comparison_payload["embedding_analysis"]
    hyb = comparison_payload["hybrid_ablation"]
    lines.extend([
        "",
        "## 2. Temporal Customer Embeddings & Sequence vs XGBoost",
        "",
        f"- **Sequence Embedding Dim**: `{emb['embedding_dim']}` | **Archetype Linear Probe Accuracy**: `{emb['archetype_linear_probe_accuracy']:.4f}` | **Recent-3 Step Attention Mass**: `{emb['mean_recent3_attention_mass']:.3f}`",
        f"- **Hybrid Ablation (XGBoost + Temporal Embeddings vs XGBoost Tabular)**: `Delta ROC-AUC = {hyb['delta_roc_auc']:+.5f}`, `Delta PR-AUC = {hyb['delta_pr_auc']:+.5f}`, `Delta MAE = {hyb['delta_mae']:+.4f}` days",
        "",
        "## 3. Customer Payment Behavior Clustering (K-Means + DBSCAN)",
        "",
        f"- **K-Means Alignment vs Latent Archetypes**: `ARI = {clustering_payload['alignment_metrics']['kmeans_ari_vs_true_archetype']:.4f}`, `NMI = {clustering_payload['alignment_metrics']['kmeans_nmi_vs_true_archetype']:.4f}`, `Silhouette = {clustering_payload['alignment_metrics']['kmeans_silhouette']:.4f}`",
        f"- **DBSCAN Density Outliers**: `{clustering_payload['dbscan_summary']['n_outliers']}` customers (`{clustering_payload['dbscan_summary']['outlier_pct']}%`) with mean late rate `{clustering_payload['dbscan_summary']['outlier_mean_late_rate']:.3f}` vs inliers `{clustering_payload['dbscan_summary']['inlier_mean_late_rate']:.3f}`",
        "",
        "| Segment Name | Customers | Mean Delay (d) | Delay Slope | Late Rate | Partial Pay Rate | Avg Util |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for p in clustering_payload["cluster_profiles"]:
        lines.append(
            f"| {p['segment_name']} | {p['n_customers']} ({p['share_pct']}%) | {p['mean_delay_days']:.1f} | "
            f"{p['delay_slope']:.3f} | {p['late_rate']:.3f} | {p['partial_payment_rate']:.3f} | {p['avg_balance_utilization']:.3f} |"
        )

    lines.extend([
        "",
        "## 4. Collection Action Optimization (Contextual Bandit Simulation)",
        "",
        f"> **Disclaimer**: {bandit_payload['disclaimer']}",
        "",
        "| Policy | Mean Cost ($) | Est. Pay Prob | OPE IPS | OPE SNIPS | OPE Doubly Robust | Sim Oracle Reward |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in bandit_payload["policy_comparison"]:
        lines.append(
            f"| {row['policy']} | {row['mean_action_cost']:.2f} | {row['estimated_payment_prob_model']:.4f} | "
            f"{row['ope_ips_reward']:.2f} | {row['ope_snips_reward']:.2f} | {row['ope_doubly_robust_reward']:.2f} | "
            f"{row['sim_oracle_expected_reward']:.2f} |"
        )

    lines.extend([
        "",
        "## 5. SHAP Explainability Highlights",
        "",
        f"- **High-Risk Customers Identified**: `{shap_summary['n_high_risk_customers_identified']}` (threshold >= `{shap_summary['high_risk_threshold']}`), explained `{shap_summary['n_high_risk_customers_explained']}`",
        "- **Top 5 Global SHAP Drivers (XGBoost)**: " + ", ".join(
            f"`{r['feature']}` ({r['xgb_mean_abs_shap']:.4f})"
            for r in shap_summary["global_feature_importance"][:5]
        ),
        "",
        "## 6. Robustness Summary",
        "",
        "See `reports/robustness_report.md` for full 5-scenario stress test matrix and model weakness diagnoses.",
        "",
    ])
    for w in robustness_payload["weakness_summary"]:
        lines.append(
            f"- **{w['model_display']}**: worst AUC drop `{w['worst_delta_roc_auc']:+.4f}` ({w['worst_roc_auc_scenario']}), "
            f"worst ECE drift `{w['worst_delta_ece']:+.4f}` ({w['worst_calibration_scenario']}), "
            f"worst MAE jump `{w['worst_delta_mae']:+.2f}d`"
        )
    return "\n".join(lines)


def run_full_experiment(
    config: ExperimentConfig,
    base_dir: str | Path = ".",
) -> dict[str, Any]:
    """Execute the complete reproducible ML research workflow and save all versioned artifacts."""
    set_global_seed(config.seed)
    tracker = ExperimentTracker(config=config, base_dir=base_dir)
    tracker.save_config_and_env()

    # 1. Longitudinal simulation & leakage-free temporal dataset prep
    customers_df, transactions_df = generate_receivables_dataset(config.data, seed=config.seed)
    datasets = prepare_datasets(transactions=transactions_df, config=config.data)

    dataset_summary = {
        "n_customers": int(len(customers_df)),
        "n_total_transactions": int(len(transactions_df)),
        "n_train": int(len(datasets.train.y_late)),
        "n_val": int(len(datasets.val.y_late)),
        "n_test": int(len(datasets.test.y_late)),
        "train_positive_rate": round(float(np.mean(datasets.train.y_late)), 4),
        "val_positive_rate": round(float(np.mean(datasets.val.y_late)), 4),
        "test_positive_rate": round(float(np.mean(datasets.test.y_late)), 4),
        "split_cutoffs": datasets.split_cutoffs,
        "n_seq_features": len(datasets.seq_feature_names),
        "n_ctx_features": len(datasets.ctx_feature_names),
        "n_tab_features": len(datasets.tab_feature_names),
        "seq_feature_names": datasets.seq_feature_names,
        "tab_feature_names": datasets.tab_feature_names,
        "archetype_counts": customers_df["true_archetype"].value_counts().to_dict(),
    }

    # 2. Train & compare Logistic Regression, Random Forest, XGBoost, and Sequence Model
    comparison = run_model_comparison(datasets=datasets, config=config)

    # 3. Customer Payment Behavior Clustering (K-Means + DBSCAN + Segmentation Lift)
    clustering = fit_clustering_and_segmentation(
        datasets=datasets,
        clustering_cfg=config.clustering,
        tabular_cfg=config.tabular,
        seed=config.seed,
    )

    # 4. Collection Action Optimization (Contextual Bandit Simulation)
    bandit_report = run_collection_bandit_simulation(
        customer_pool_meta=datasets.test.meta,
        config=config.bandit,
        seed=config.seed,
    )

    # 5. Model Explainability (SHAP global + every high-risk customer)
    shap_results = generate_shap_explanations(
        xgb_model=comparison.tabular_models["xgboost"],  # type: ignore[arg-type]
        rf_model=comparison.tabular_models["random_forest"],  # type: ignore[arg-type]
        logreg_model=comparison.tabular_models["logistic_regression"],  # type: ignore[arg-type]
        seq_model=comparison.sequence_model,
        train_ds=datasets.train,
        test_ds=datasets.test,
        tab_feature_names=datasets.tab_feature_names,
        xgb_test_probs=comparison.test_predictions["xgboost"].probs,
        seq_test_probs=comparison.test_predictions["sequence_model"].probs,
        config=config.explainability,
        seed=config.seed,
    )

    # 6. Robustness Stress Testing (5 shift/corruption scenarios)
    robustness_results = run_robustness_suite(
        datasets=datasets,
        comparison=comparison,
        config=config,
    )

    # 7. Save Models, Figures, Reports, and Registry
    model_paths = tracker.save_models(comparison=comparison, clustering=clustering)
    figure_paths = tracker.generate_figures(
        comparison=comparison,
        clustering=clustering,
        bandit=bandit_report,
        shap_results=shap_results,
        robustness=robustness_results,
    )

    comparison_payload = {
        "thresholds": comparison.thresholds,
        "val_metrics": comparison.val_metrics,
        "test_metrics": comparison.test_metrics,
        "curve_data": comparison.curve_data,
        "sequence_vs_xgboost": comparison.sequence_vs_xgboost,
        "embedding_analysis": comparison.embedding_analysis,
        "hybrid_ablation": comparison.hybrid_ablation,
        "sequence_train_history": comparison.sequence_model.train_history,
    }
    clustering_payload = {
        "kmeans_k_sweep": clustering.kmeans_k_sweep,
        "dbscan_eps_sweep": clustering.dbscan_eps_sweep,
        "cluster_profiles": clustering.cluster_profiles,
        "dbscan_summary": clustering.dbscan_summary,
        "alignment_metrics": clustering.alignment_metrics,
        "segmentation_lift": clustering.segmentation_lift,
    }
    bandit_payload = asdict(bandit_report)
    shap_summary = {
        k: v for k, v in shap_results.items() if k != "high_risk_customer_explanations"
    }
    robustness_payload = {
        k: v for k, v in robustness_results.items() if k != "markdown_report"
    }

    metrics_payload = {
        "run_id": tracker.run_id,
        "experiment_name": config.experiment_name,
        "model_version": config.model_version,
        "config_hash": config.config_hash(),
        "dataset_summary": dataset_summary,
        "model_comparison": comparison_payload,
        "clustering": clustering_payload,
        "bandit": bandit_payload,
        "explainability_summary": shap_summary,
        "robustness": robustness_payload,
    }

    summary_md = _build_experiment_summary_markdown(
        run_id=tracker.run_id,
        config=config,
        dataset_summary=dataset_summary,
        comparison_payload=comparison_payload,
        clustering_payload=clustering_payload,
        bandit_payload=bandit_payload,
        shap_summary=shap_summary,
        robustness_payload=robustness_payload,
    )

    run_dir = tracker.finalize_run(
        metrics_payload=metrics_payload,
        high_risk_explanations=shap_results["high_risk_customer_explanations"],
        summary_markdown=summary_md,
        robustness_markdown=robustness_results["markdown_report"],
        model_paths=model_paths,
        figure_paths=figure_paths,
    )

    return {
        "run_id": tracker.run_id,
        "run_dir": str(run_dir),
        "metrics": metrics_payload,
        "n_high_risk_explained": len(shap_results["high_risk_customer_explanations"]),
    }
