"""Reproducible experiment tracking, artifact versioning, and model registry."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from receivables_ml.config import ExperimentConfig, capture_environment_metadata
from receivables_ml.evaluation.comparison import MODEL_DISPLAY_NAMES


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return p


class ExperimentTracker:
    """Manages versioned experiment run directories, figures, models, and registry."""

    def __init__(self, config: ExperimentConfig, base_dir: str | Path = ".") -> None:
        self.config = config
        self.base_dir = Path(base_dir)
        self.output_root = self.base_dir / config.output_root
        self.runs_root = self.output_root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)

        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{stamp}_{config.experiment_name}_v{config.model_version}_{config.config_hash()}"
        self.run_dir = self.runs_root / self.run_id
        self.models_dir = self.run_dir / "models"
        self.figures_dir = self.run_dir / "figures"
        self.reports_dir = self.run_dir / "reports"
        for d in (self.models_dir, self.figures_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save_config_and_env(self) -> None:
        cfg_dict = self.config.model_dump(mode="json")
        cfg_dict["config_hash"] = self.config.config_hash()
        save_json(self.run_dir / "config.json", cfg_dict)

        try:
            import yaml

            (self.run_dir / "config.yaml").write_text(
                yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8"
            )
        except ImportError:
            pass

        env = capture_environment_metadata()
        env["run_id"] = self.run_id
        env["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save_json(self.run_dir / "environment.json", env)

    def save_models(self, comparison: Any, clustering: Any) -> dict[str, str]:
        paths: dict[str, str] = {}
        for name, model in comparison.tabular_models.items():
            p = model.save(self.models_dir)
            paths[name] = str(p.relative_to(self.run_dir))
        seq_p = comparison.sequence_model.save(self.models_dir)
        paths["sequence_model"] = str(seq_p.relative_to(self.run_dir))

        clust_p = self.models_dir / "clustering.joblib"
        joblib.dump(
            {
                "scaler": clustering.scaler,
                "kmeans": clustering.kmeans,
                "dbscan": clustering.dbscan,
                "dbscan_eps": clustering.dbscan_eps,
                "persona_map": clustering.persona_map,
            },
            clust_p,
        )
        paths["clustering"] = str(clust_p.relative_to(self.run_dir))
        return paths

    def generate_figures(
        self,
        comparison: Any,
        clustering: Any,
        bandit: Any,
        shap_results: dict[str, Any],
        robustness: dict[str, Any],
    ) -> list[str]:
        sns.set_theme(style="whitegrid", font_scale=0.95)
        saved: list[str] = []

        # 1. Model Comparison Bar Chart (ROC-AUC, PR-AUC, F1, 1-ECE, MAE)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        models = list(comparison.test_metrics.keys())
        labels = [MODEL_DISPLAY_NAMES[m] for m in models]

        clf_df = pd.DataFrame(
            [
                {
                    "Model": MODEL_DISPLAY_NAMES[m],
                    "ROC-AUC": comparison.test_metrics[m]["roc_auc"],
                    "PR-AUC": comparison.test_metrics[m]["pr_auc"],
                    "F1": comparison.test_metrics[m]["f1"],
                }
                for m in models
            ]
        ).melt(id_vars="Model", var_name="Metric", value_name="Score")
        sns.barplot(data=clf_df, x="Model", y="Score", hue="Metric", ax=axes[0])
        axes[0].set_title("Classification Discrimination & F1 (Test)")
        axes[0].set_ylim(0.0, 1.05)
        axes[0].tick_params(axis="x", rotation=15)

        reg_df = pd.DataFrame(
            [
                {
                    "Model": MODEL_DISPLAY_NAMES[m],
                    "MAE (days)": comparison.test_metrics[m]["mae"],
                    "RMSE (days)": comparison.test_metrics[m]["rmse"],
                }
                for m in models
            ]
        ).melt(id_vars="Model", var_name="Metric", value_name="Days")
        sns.barplot(data=reg_df, x="Model", y="Days", hue="Metric", ax=axes[1])
        axes[1].set_title("Payment Delay Regression Error (Test)")
        axes[1].tick_params(axis="x", rotation=15)
        plt.tight_layout()
        p1 = self.figures_dir / "model_comparison_metrics.png"
        fig.savefig(p1, dpi=160)
        plt.close(fig)
        saved.append(str(p1.relative_to(self.run_dir)))

        # 2. ROC, PR, and Calibration Reliability Curves
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for m in models:
            c = comparison.curve_data[m]
            disp = MODEL_DISPLAY_NAMES[m]
            axes[0].plot(c["fpr"], c["tpr"], label=f"{disp} ({comparison.test_metrics[m]['roc_auc']:.3f})")
            axes[1].plot(c["recall"], c["precision"], label=f"{disp} ({comparison.test_metrics[m]['pr_auc']:.3f})")

            bins = comparison.test_metrics[m]["calibration_bins"]
            xs = [b["mean_pred_prob"] for b in bins if b["empirical_pos_rate"] is not None]
            ys = [b["empirical_pos_rate"] for b in bins if b["empirical_pos_rate"] is not None]
            axes[2].plot(xs, ys, marker="o", label=f"{disp} (ECE={comparison.test_metrics[m]['calibration_ece']:.3f})")

        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[0].set_title("ROC Curves")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].legend(fontsize=8)

        axes[1].set_title("Precision-Recall Curves")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].legend(fontsize=8)

        axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[2].set_title("Calibration Reliability Diagram")
        axes[2].set_xlabel("Mean Predicted Probability")
        axes[2].set_ylabel("Empirical Late Rate")
        axes[2].legend(fontsize=8)
        plt.tight_layout()
        p2 = self.figures_dir / "roc_pr_calibration_curves.png"
        fig.savefig(p2, dpi=160)
        plt.close(fig)
        saved.append(str(p2.relative_to(self.run_dir)))

        # 3. Temporal Customer Embeddings PCA Projection
        proj_df = pd.DataFrame(comparison.embedding_analysis["customer_projection"])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=proj_df, x="pca_1", y="pca_2", hue="true_archetype", alpha=0.82, s=48, ax=ax)
        ax.set_title("Temporal Customer Embeddings (PCA Projection of GRU+Attn Representation)")
        ax.legend(title="True Behavior Archetype", fontsize=8)
        plt.tight_layout()
        p3 = self.figures_dir / "temporal_embeddings_pca.png"
        fig.savefig(p3, dpi=160)
        plt.close(fig)
        saved.append(str(p3.relative_to(self.run_dir)))

        # 4. Customer Payment Behavior Clustering (Delay Mean vs Delay Slope)
        beh_df = clustering.train_customer_assignments
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        sns.scatterplot(
            data=beh_df, x="delay_mean", y="delay_slope", hue="behavior_segment",
            size="log_amount_mean", sizes=(25, 160), alpha=0.8, ax=axes[0],
        )
        axes[0].set_title("K-Means Customer Behavior Segments")
        axes[0].set_xlabel("Mean Historical Payment Delay (days)")
        axes[0].set_ylabel("Payment Delay Trajectory Slope")
        axes[0].legend(fontsize=7.5)

        beh_df_plot = beh_df.copy()
        beh_df_plot["DBSCAN Status"] = np.where(beh_df_plot["is_dbscan_outlier"] == 1, "Outlier / Anomaly (-1)", "Core Cluster")
        sns.scatterplot(
            data=beh_df_plot, x="delay_mean", y="delay_std", hue="DBSCAN Status", alpha=0.82, ax=axes[1]
        )
        axes[1].set_title("DBSCAN Density Clustering & Behavioral Outliers")
        axes[1].set_xlabel("Mean Historical Payment Delay (days)")
        axes[1].set_ylabel("Payment Delay Std Dev (Volatility)")
        plt.tight_layout()
        p4 = self.figures_dir / "customer_behavior_clusters.png"
        fig.savefig(p4, dpi=160)
        plt.close(fig)
        saved.append(str(p4.relative_to(self.run_dir)))

        # 5. Contextual Bandit Off-Policy Reward Comparison
        pol_df = pd.DataFrame(bandit.policy_comparison)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=pol_df, y="policy", x="ope_doubly_robust_reward", color="#2a7fff", ax=ax)
        ax.set_title("Simulated Collection Policy Evaluation (Doubly Robust OPE Reward - Non-Causal)")
        ax.set_xlabel("Estimated Expected Net Collection Reward (Simulated Units)")
        ax.set_ylabel("")
        plt.tight_layout()
        p5 = self.figures_dir / "bandit_policy_comparison.png"
        fig.savefig(p5, dpi=160)
        plt.close(fig)
        saved.append(str(p5.relative_to(self.run_dir)))

        # 6. Global SHAP Importance Top 15 Features
        shap_df = pd.DataFrame(shap_results["global_feature_importance"][:15])
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        sns.barplot(data=shap_df, y="feature", x="xgb_mean_abs_shap", hue="feature_group", dodge=False, ax=ax)
        ax.set_title("Global SHAP Feature Importance (XGBoost TreeExplainer)")
        ax.set_xlabel("Mean |SHAP Value| (Log-Odds Impact)")
        ax.set_ylabel("")
        plt.tight_layout()
        p6 = self.figures_dir / "shap_global_importance.png"
        fig.savefig(p6, dpi=160)
        plt.close(fig)
        saved.append(str(p6.relative_to(self.run_dir)))

        # 7. Robustness Stress-Test ROC-AUC Delta Heatmap
        rob_df = pd.DataFrame(robustness["stress_test_results"])
        piv = rob_df.pivot(index="scenario_label", columns="model_display", values="delta_roc_auc")
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        sns.heatmap(piv, annot=True, fmt=".3f", cmap="RdYlGn", center=0.0, ax=ax)
        ax.set_title("Robustness Stress Testing: ROC-AUC Change Under Distribution Shifts")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=18)
        plt.tight_layout()
        p7 = self.figures_dir / "robustness_stress_heatmap.png"
        fig.savefig(p7, dpi=160)
        plt.close(fig)
        saved.append(str(p7.relative_to(self.run_dir)))

        return saved

    def finalize_run(
        self,
        metrics_payload: dict[str, Any],
        high_risk_explanations: list[dict[str, Any]],
        summary_markdown: str,
        robustness_markdown: str,
        model_paths: dict[str, str],
        figure_paths: list[str],
    ) -> Path:
        save_json(self.run_dir / "metrics.json", metrics_payload)
        save_json(self.reports_dir / "high_risk_explanations.json", high_risk_explanations)
        (self.reports_dir / "experiment_summary.md").write_text(summary_markdown, encoding="utf-8")
        (self.reports_dir / "robustness_report.md").write_text(robustness_markdown, encoding="utf-8")

        registry_path = self.output_root / "registry.json"
        if registry_path.exists():
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        else:
            registry = {"runs": []}

        entry = {
            "run_id": self.run_id,
            "experiment_name": self.config.experiment_name,
            "model_version": self.config.model_version,
            "config_hash": self.config.config_hash(),
            "seed": self.config.seed,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_dir": str(self.run_dir),
            "model_paths": model_paths,
            "figure_paths": figure_paths,
            "test_summary": {
                m: {
                    k: round(float(v), 4)
                    for k, v in vals.items()
                    if k in {"roc_auc", "pr_auc", "f1", "calibration_ece", "mae", "rmse"}
                }
                for m, vals in metrics_payload["model_comparison"]["test_metrics"].items()
            },
        }
        registry["runs"].append(entry)
        save_json(registry_path, registry)
        (self.output_root / "latest_run.txt").write_text(str(self.run_dir.resolve()), encoding="utf-8")
        return self.run_dir
