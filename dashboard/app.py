"""Interactive ReceivablesOS ML Model Comparison & Research Intelligence Dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "sequence_model": "Sequence Model (GRU+Attn)",
}

st.set_page_config(
    page_title="ReceivablesOS ML Intelligence & Model Comparison",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _discover_runs() -> list[dict[str, Any]]:
    registry_path = ARTIFACTS_DIR / "registry.json"
    if not registry_path.exists():
        return []
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return list(reversed(data.get("runs", [])))


@st.cache_data
def _load_run_bundle(run_dir_str: str) -> dict[str, Any]:
    run_dir = Path(run_dir_str)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    hr_path = run_dir / "reports" / "high_risk_explanations.json"
    high_risk = json.loads(hr_path.read_text(encoding="utf-8")) if hr_path.exists() else []
    rob_md_path = run_dir / "reports" / "robustness_report.md"
    rob_md = rob_md_path.read_text(encoding="utf-8") if rob_md_path.exists() else ""
    return {
        "metrics": metrics,
        "config": config,
        "env": env,
        "high_risk": high_risk,
        "robustness_md": rob_md,
    }


def main() -> None:
    st.title("ReceivablesOS — ML Research & Model Comparison Dashboard")
    runs = _discover_runs()
    if not runs:
        st.error(
            "No saved experiment runs found in `artifacts/registry.json`. "
            "Run `python scripts/run_experiments.py` first."
        )
        return

    with st.sidebar:
        st.header("Experiment Version Selector")
        run_options = {f"{r['run_id']}": r["run_dir"] for r in runs}
        selected_id = st.selectbox("Experiment Run", list(run_options.keys()))
        bundle = _load_run_bundle(run_options[selected_id])
        m = bundle["metrics"]
        st.caption(f"Model Version: `{m['model_version']}`")
        st.caption(f"Config Hash: `{m['config_hash']}` | Seed: `{bundle['config']['seed']}`")
        ds = m["dataset_summary"]
        st.metric("Customers", ds["n_customers"])
        st.metric("Test Invoices", ds["n_test"])
        st.metric("Test Late Rate", f"{100 * ds['test_positive_rate']:.1f}%")

    tabs = st.tabs(
        [
            "Model Comparison",
            "Temporal Embeddings",
            "Behavior Clustering",
            "Collection Bandit",
            "SHAP Risk Inspector",
            "Robustness & Weaknesses",
            "Reproducibility",
        ]
    )

    comp = m["model_comparison"]
    test_metrics = comp["test_metrics"]

    # Tab 1: Model Comparison
    with tabs[0]:
        st.subheader("Holdout Test Benchmark: Logistic Regression vs Random Forest vs XGBoost vs Sequence Model")
        rows = []
        for k, vals in test_metrics.items():
            rows.append(
                {
                    "Model": MODEL_LABELS.get(k, k),
                    "ROC-AUC": round(vals["roc_auc"], 4),
                    "PR-AUC": round(vals["pr_auc"], 4),
                    "Precision": round(vals["precision"], 4),
                    "Recall": round(vals["recall"], 4),
                    "F1": round(vals["f1"], 4),
                    "Calibration ECE": round(vals["calibration_ece"], 4),
                    "Brier Score": round(vals["brier_score"], 4),
                    "Delay MAE (days)": round(vals["mae"], 2),
                    "Delay RMSE (days)": round(vals["rmse"], 2),
                    "Val Threshold": round(vals["threshold"], 2),
                }
            )
        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df, width="stretch", hide_index=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            fig_roc = go.Figure()
            for k, cdata in comp["curve_data"].items():
                fig_roc.add_trace(
                    go.Scatter(
                        x=cdata["fpr"], y=cdata["tpr"], mode="lines",
                        name=f"{MODEL_LABELS.get(k, k)} ({test_metrics[k]['roc_auc']:.3f})",
                    )
                )
            fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
            fig_roc.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR", height=370)
            st.plotly_chart(fig_roc, width="stretch")

        with c2:
            fig_pr = go.Figure()
            for k, cdata in comp["curve_data"].items():
                fig_pr.add_trace(
                    go.Scatter(
                        x=cdata["recall"], y=cdata["precision"], mode="lines",
                        name=f"{MODEL_LABELS.get(k, k)} ({test_metrics[k]['pr_auc']:.3f})",
                    )
                )
            fig_pr.update_layout(title="Precision-Recall Curves", xaxis_title="Recall", yaxis_title="Precision", height=370)
            st.plotly_chart(fig_pr, width="stretch")

        with c3:
            fig_cal = go.Figure()
            for k, vals in test_metrics.items():
                bins = [b for b in vals["calibration_bins"] if b["empirical_pos_rate"] is not None]
                fig_cal.add_trace(
                    go.Scatter(
                        x=[b["mean_pred_prob"] for b in bins],
                        y=[b["empirical_pos_rate"] for b in bins],
                        mode="lines+markers",
                        name=f"{MODEL_LABELS.get(k, k)} (ECE={vals['calibration_ece']:.3f})",
                    )
                )
            fig_cal.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
            fig_cal.update_layout(title="Calibration Reliability", xaxis_title="Predicted Prob", yaxis_title="Empirical Rate", height=370)
            st.plotly_chart(fig_cal, width="stretch")

        st.markdown("#### Sequence Model vs XGBoost: Slice Performance by Customer Archetype & History Length")
        col_a, col_b = st.columns(2)
        with col_a:
            st.dataframe(pd.DataFrame(comp["sequence_vs_xgboost"]["by_archetype"]), width="stretch", hide_index=True)
        with col_b:
            st.dataframe(pd.DataFrame(comp["sequence_vs_xgboost"]["by_history_length"]), width="stretch", hide_index=True)

    # Tab 2: Temporal Customer Embeddings
    with tabs[1]:
        st.subheader("Temporal Customer Embeddings (PyTorch GRU + Temporal Attention)")
        emb = comp["embedding_analysis"]
        hyb = comp["hybrid_ablation"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Embedding Dimension", emb["embedding_dim"])
        k2.metric("Archetype Probe Acc", f"{100 * emb['archetype_linear_probe_accuracy']:.1f}%")
        k3.metric("Recent-3 Attention Mass", f"{100 * emb['mean_recent3_attention_mass']:.1f}%")
        k4.metric("Hybrid XGB+Emb Delta AUC", f"{hyb['delta_roc_auc']:+.4f}")

        proj_df = pd.DataFrame(emb["customer_projection"])
        fig_emb = px.scatter(
            proj_df,
            x="pca_1",
            y="pca_2",
            color="true_archetype",
            hover_data=["customer_id", "hist_delay_mean", "balance_utilization"],
            title="PCA Projection of Learned Temporal Customer Embeddings",
            height=480,
        )
        st.plotly_chart(fig_emb, width="stretch")

        st.markdown("#### Sequence Signals & Training Convergence")
        hist_df = pd.DataFrame(comp["sequence_train_history"])
        if not hist_df.empty:
            fig_loss = px.line(hist_df, x="epoch", y=["train_loss", "val_loss"], title="Sequence Model Multi-Task Loss Curve")
            st.plotly_chart(fig_loss, width="stretch")

    # Tab 3: Customer Behavior Clustering
    with tabs[2]:
        st.subheader("Customer Payment Behavior Clustering (K-Means & DBSCAN)")
        cl = m["clustering"]
        c1, c2, c3 = st.columns(3)
        c1.metric("K-Means Silhouette", f"{cl['alignment_metrics']['kmeans_silhouette']:.3f}")
        c2.metric("K-Means ARI vs Archetype", f"{cl['alignment_metrics']['kmeans_ari_vs_true_archetype']:.3f}")
        c3.metric("DBSCAN Outliers", f"{cl['dbscan_summary']['n_outliers']} ({cl['dbscan_summary']['outlier_pct']}%)")

        st.markdown("#### Discovered K-Means Customer Behavior Personas")
        st.dataframe(pd.DataFrame(cl["cluster_profiles"]), width="stretch", hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            sweep_df = pd.DataFrame(cl["kmeans_k_sweep"])
            fig_k = px.line(sweep_df, x="k", y="silhouette", markers=True, title="K-Means Silhouette Across k")
            st.plotly_chart(fig_k, width="stretch")
        with col2:
            lift_rows = []
            for model_k, vals in cl["segmentation_lift"]["model_feature_lift"].items():
                lift_rows.append({"Model": MODEL_LABELS.get(model_k, model_k), **vals})
            st.markdown("#### Downstream Segmentation Feature Lift")
            st.dataframe(pd.DataFrame(lift_rows), width="stretch", hide_index=True)
            st.json(cl["dbscan_summary"])

    # Tab 4: Contextual Bandit Simulation
    with tabs[3]:
        st.subheader("Collection Action Optimization (Contextual Bandit Simulation)")
        bandit = m["bandit"]
        st.warning(bandit["disclaimer"])

        pol_df = pd.DataFrame(bandit["policy_comparison"])
        st.dataframe(pol_df, width="stretch", hide_index=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig_pol = px.bar(
                pol_df,
                y="policy",
                x="ope_doubly_robust_reward",
                error_x="ope_dr_std_err",
                orientation="h",
                title="Doubly-Robust Off-Policy Expected Reward (Simulated)",
            )
            st.plotly_chart(fig_pol, width="stretch")
        with col_b2:
            seg_df = pd.DataFrame(bandit["action_probability_by_segment"])
            fig_seg = px.bar(
                seg_df,
                x="customer_archetype",
                y="estimated_payment_prob",
                color="action",
                barmode="group",
                title="Estimated Payment Probability by Segment & Action (Simulated)",
            )
            st.plotly_chart(fig_seg, width="stretch")

    # Tab 5: SHAP High-Risk Customer Explainability
    with tabs[4]:
        st.subheader("SHAP Model Explainability & High-Risk Customer Diagnostics")
        shap_sum = m["explainability_summary"]
        g_df = pd.DataFrame(shap_sum["global_feature_importance"][:15])
        fig_shap = px.bar(
            g_df[::-1],
            x="xgb_mean_abs_shap",
            y="feature",
            color="feature_group",
            orientation="h",
            title="Top 15 Global Risk Drivers (XGBoost TreeExplainer Mean |SHAP|)",
            height=430,
        )
        st.plotly_chart(fig_shap, width="stretch")

        st.markdown("#### Per-Customer High-Risk Explanation Drilldown")
        hr_list = bundle["high_risk"]
        if hr_list:
            options = [
                f"{item['customer_id']} | {item['invoice_id']} | XGB={item['xgb_risk_prob']:.2f} | Seq={item['sequence_risk_prob']:.2f} | {item['industry']}"
                for item in hr_list
            ]
            chosen_label = st.selectbox("Select High-Risk Customer", options)
            item = hr_list[options.index(chosen_label)]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("XGBoost Risk Prob", f"{item['xgb_risk_prob']:.3f}")
            c2.metric("Sequence Risk Prob", f"{item['sequence_risk_prob']:.3f}")
            c3.metric("Outstanding Balance SHAP", f"{item['outstanding_balance_contribution']['total_shap_log_odds']:+.3f}")
            c4.metric("Previous Delays SHAP", f"{item['previous_payment_delays_contribution']['total_shap_log_odds']:+.3f}")

            left, right = st.columns(2)
            with left:
                fac_df = pd.DataFrame(item["most_influential_risk_factors"])
                fig_f = px.bar(
                    fac_df[::-1],
                    x="shap_contribution",
                    y="feature",
                    color="direction",
                    orientation="h",
                    title="Most Influential Risk Factors (SHAP Log-Odds)",
                )
                st.plotly_chart(fig_f, width="stretch")
            with right:
                trends = item["historical_payment_trends"]
                tr_df = pd.DataFrame(
                    {
                        "Historical Invoice Step": list(range(1, len(trends["recent_delays_days"]) + 1)),
                        "Observed Delay (days)": trends["recent_delays_days"],
                        "Outstanding Balance Before ($)": trends["recent_outstanding_balances"],
                    }
                )
                fig_tr = px.line(
                    tr_df,
                    x="Historical Invoice Step",
                    y="Observed Delay (days)",
                    markers=True,
                    title="Historical Payment Delay Trajectory",
                )
                st.plotly_chart(fig_tr, width="stretch")
                st.json(
                    {
                        "outstanding_balance_contribution": item["outstanding_balance_contribution"],
                        "previous_payment_delays_contribution": item["previous_payment_delays_contribution"],
                        "sequence_attention_weights": trends["sequence_attention_weights"],
                    }
                )

    # Tab 6: Robustness Testing
    with tabs[5]:
        st.subheader("Robustness Stress Testing & Model Weakness Report")
        rob = m["robustness"]
        rob_df = pd.DataFrame(rob["stress_test_results"])
        piv = rob_df.pivot(index="scenario_label", columns="model_display", values="delta_roc_auc")
        fig_hm = px.imshow(
            piv,
            text_auto=".3f",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0.0,
            title="ROC-AUC Change Under Distribution Shifts & Data Corruption",
            height=460,
        )
        st.plotly_chart(fig_hm, width="stretch")
        st.dataframe(rob_df, width="stretch", hide_index=True)
        with st.expander("Full Diagnostic Robustness & Weakness Report", expanded=True):
            st.markdown(bundle["robustness_md"])

    # Tab 7: Reproducibility
    with tabs[6]:
        st.subheader("Reproducibility Metadata, Configuration & Model Registry")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### Frozen Experiment Configuration")
            st.json(bundle["config"])
        with col_r2:
            st.markdown("#### Runtime Environment & Package Versions")
            st.json(bundle["env"])


if __name__ == "__main__":
    main()
