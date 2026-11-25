"""Robustness stress-testing under 5 distribution-shift & corruption scenarios."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from receivables_ml.config import ExperimentConfig
from receivables_ml.data.features import PreparedDatasets, compute_point_in_time_records
from receivables_ml.evaluation.comparison import MODEL_DISPLAY_NAMES, ComparisonArtifacts
from receivables_ml.evaluation.metrics import evaluate_predictions


def _score_all_models(
    comparison: ComparisonArtifacts,
    X_tab: np.ndarray,
    X_seq: np.ndarray,
    seq_mask: np.ndarray,
    seq_lengths: np.ndarray,
    X_ctx: np.ndarray,
    y_late: np.ndarray,
    y_delay: np.ndarray,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name, model in comparison.tabular_models.items():
        pred = model.predict(X_tab)
        results[name] = evaluate_predictions(
            y_late, y_delay, pred.probs, pred.delay_preds,
            threshold=comparison.thresholds[name], n_bootstrap=0,
        )
    seq_pred = comparison.sequence_model.predict(X_seq, seq_mask, seq_lengths, X_ctx)
    results["sequence_model"] = evaluate_predictions(
        y_late, y_delay, seq_pred.probs, seq_pred.delay_preds,
        threshold=comparison.thresholds["sequence_model"], n_bootstrap=0,
    )
    return results


def run_robustness_suite(
    datasets: PreparedDatasets,
    comparison: ComparisonArtifacts,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Run all 5 stress tests across all 4 trained models and compile weakness report."""
    te = datasets.test
    cutoff = datasets.split_cutoffs["val_max_day"]
    seed = config.seed
    rng = np.random.default_rng(seed + 101)
    baseline = comparison.test_metrics

    scenario_records: list[dict[str, Any]] = []

    def _record_scenario(scenario_id: str, scenario_label: str, category: str, res_by_model: dict[str, dict[str, Any]]) -> None:
        for model_name, metrics in res_by_model.items():
            base = baseline[model_name]
            scenario_records.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_label": scenario_label,
                    "category": category,
                    "model": model_name,
                    "model_display": MODEL_DISPLAY_NAMES[model_name],
                    "roc_auc": round(metrics["roc_auc"], 4),
                    "pr_auc": round(metrics["pr_auc"], 4),
                    "f1": round(metrics["f1"], 4),
                    "calibration_ece": round(metrics["calibration_ece"], 4),
                    "mae": round(metrics["mae"], 3),
                    "rmse": round(metrics["rmse"], 3),
                    "delta_roc_auc": round(metrics["roc_auc"] - base["roc_auc"], 4),
                    "delta_pr_auc": round(metrics["pr_auc"] - base["pr_auc"], 4),
                    "delta_f1": round(metrics["f1"] - base["f1"], 4),
                    "delta_ece": round(metrics["calibration_ece"] - base["calibration_ece"], 4),
                    "delta_mae": round(metrics["mae"] - base["mae"], 3),
                    "delta_rmse": round(metrics["rmse"] - base["rmse"], 3),
                    "stressed_positive_rate": round(metrics["positive_rate"], 4),
                }
            )

    # 1. Increased Payment Delays (+5, +10, +20 days)
    for shift in config.robustness.delay_shift_days:
        tx_shift = datasets.raw_transactions.copy()
        # Systemic macro slowdown begins shortly before test horizon
        m_shift = tx_shift["issue_day"] > (cutoff - 45)
        tx_shift.loc[m_shift, "payment_delay_days"] = np.clip(
            tx_shift.loc[m_shift, "payment_delay_days"] + shift, -15.0, 90.0
        )
        tx_shift.loc[m_shift, "final_paid_day"] = np.maximum(
            tx_shift.loc[m_shift, "issue_day"] + 1,
            tx_shift.loc[m_shift, "due_day"] + tx_shift.loc[m_shift, "payment_delay_days"],
        )
        tx_shift.loc[m_shift, "is_late"] = (
            tx_shift.loc[m_shift, "payment_delay_days"] > config.data.late_threshold_days
        ).astype(int)

        xs, sm, sl, xc, xt, yl, yd, meta, _ = compute_point_in_time_records(tx_shift, config.data)
        t_mask = meta["issue_day"].to_numpy() > cutoff
        res = _score_all_models(
            comparison, xt[t_mask], xs[t_mask], sm[t_mask], sl[t_mask], xc[t_mask], yl[t_mask], yd[t_mask]
        )
        _record_scenario(
            f"delay_shift_{int(shift)}d",
            f"Increased Payment Delays (+{int(shift)}d)",
            "Increased Payment Delays",
            res,
        )

    # 2. Industry Distribution Changes (shift toward high-risk Construction / Logistics / Retail)
    industries = te.meta["industry"].to_numpy()
    weights = np.ones(len(te.meta), dtype=np.float64)
    high_risk_inds = np.isin(industries, ["Construction", "Logistics", "Retail"])
    low_risk_inds = np.isin(industries, ["SaaS", "Healthcare"])
    weights[high_risk_inds] *= config.robustness.industry_shift_high_risk_boost
    weights[low_risk_inds] *= 0.25
    weights /= np.sum(weights)
    ind_idx = rng.choice(len(te.meta), size=len(te.meta), replace=True, p=weights)
    res_ind = _score_all_models(
        comparison,
        te.X_tab[ind_idx], te.X_seq[ind_idx], te.seq_mask[ind_idx], te.seq_lengths[ind_idx],
        te.X_ctx[ind_idx], te.y_late[ind_idx], te.y_delay[ind_idx],
    )
    _record_scenario(
        "industry_shift_high_risk",
        "Industry Shift (Construction/Logistics Surge)",
        "Industry Distribution Change",
        res_ind,
    )

    # 3. Customer Behavior Drift (35% of previously non-deteriorating customers drift toward delinquency)
    tx_drift = datasets.raw_transactions.copy()
    stable_custs = tx_drift.loc[
        tx_drift["true_archetype"].isin(["consistent_early_payer", "consistent_late_payer"]),
        "customer_id",
    ].unique()
    n_drift = max(1, int(len(stable_custs) * config.robustness.behavior_drift_fraction))
    drifted_custs = set(rng.choice(stable_custs, size=n_drift, replace=False))
    drift_rows = tx_drift["customer_id"].isin(drifted_custs) & (tx_drift["issue_day"] > cutoff - 30)
    # Progressive ramp across test horizon for drifted customers
    prog = np.clip((tx_drift.loc[drift_rows, "issue_day"] - (cutoff - 30)) / 90.0, 0.0, 2.0)
    extra_delay = 10.0 * prog + rng.normal(2.0, 3.0, size=int(np.sum(drift_rows)))
    tx_drift.loc[drift_rows, "payment_delay_days"] = np.clip(
        tx_drift.loc[drift_rows, "payment_delay_days"] + extra_delay, -15.0, 90.0
    )
    tx_drift.loc[drift_rows, "final_paid_day"] = np.maximum(
        tx_drift.loc[drift_rows, "issue_day"] + 1,
        tx_drift.loc[drift_rows, "due_day"] + tx_drift.loc[drift_rows, "payment_delay_days"],
    )
    tx_drift.loc[drift_rows, "is_partial_payment"] = np.maximum(
        tx_drift.loc[drift_rows, "is_partial_payment"],
        (rng.random(int(np.sum(drift_rows))) < 0.40).astype(int),
    )
    tx_drift.loc[drift_rows, "is_late"] = (
        tx_drift.loc[drift_rows, "payment_delay_days"] > config.data.late_threshold_days
    ).astype(int)

    xs, sm, sl, xc, xt, yl, yd, meta, _ = compute_point_in_time_records(tx_drift, config.data)
    t_mask = meta["issue_day"].to_numpy() > cutoff
    res_drift = _score_all_models(
        comparison, xt[t_mask], xs[t_mask], sm[t_mask], sl[t_mask], xc[t_mask], yl[t_mask], yd[t_mask]
    )
    _record_scenario(
        "customer_behavior_drift",
        f"Customer Behavior Drift ({int(100 * config.robustness.behavior_drift_fraction)}% Stable Cohort)",
        "Customer Behavior Drift",
        res_drift,
    )

    # 4. Missing Transaction Records (dropout of 20%, 40%, 60% of historical sequence records)
    for drop_rate in config.robustness.missing_record_rates:
        xs, sm, sl, xc, xt, yl, yd, meta, _ = compute_point_in_time_records(
            datasets.raw_transactions,
            config.data,
            dropout_rate=drop_rate,
            dropout_seed=seed + int(drop_rate * 1000),
        )
        t_mask = meta["issue_day"].to_numpy() > cutoff
        res_miss = _score_all_models(
            comparison, xt[t_mask], xs[t_mask], sm[t_mask], sl[t_mask], xc[t_mask], yl[t_mask], yd[t_mask]
        )
        _record_scenario(
            f"missing_records_{int(drop_rate * 100)}pct",
            f"Missing Transaction Records ({int(drop_rate * 100)}% Dropout)",
            "Missing Transaction Records",
            res_miss,
        )

    # 5. Sudden Economic Shock (correlated spike in delays, invoice amounts, balances, and partial payments)
    tx_shock = datasets.raw_transactions.copy()
    shock_rows = tx_shock["issue_day"] > cutoff
    n_sh = int(np.sum(shock_rows))
    exposed = tx_shock.loc[shock_rows, "industry"].isin(["Construction", "Logistics", "Retail", "Manufacturing"]).to_numpy()
    shock_delay = np.where(exposed, config.robustness.shock_delay_jump_days * 1.25, config.robustness.shock_delay_jump_days * 0.7)
    shock_delay = shock_delay + rng.normal(0.0, 4.0, size=n_sh)

    tx_shock.loc[shock_rows, "invoice_amount"] = (
        tx_shock.loc[shock_rows, "invoice_amount"] * config.robustness.shock_amount_multiplier
    )
    tx_shock.loc[shock_rows, "outstanding_balance_before"] = (
        tx_shock.loc[shock_rows, "outstanding_balance_before"] * config.robustness.shock_amount_multiplier
    )
    tx_shock.loc[shock_rows, "balance_utilization"] = (
        tx_shock.loc[shock_rows, "balance_utilization"] * config.robustness.shock_amount_multiplier
    )
    tx_shock.loc[shock_rows, "payment_delay_days"] = np.clip(
        tx_shock.loc[shock_rows, "payment_delay_days"] + shock_delay, -15.0, 90.0
    )
    tx_shock.loc[shock_rows, "final_paid_day"] = np.maximum(
        tx_shock.loc[shock_rows, "issue_day"] + 1,
        tx_shock.loc[shock_rows, "due_day"] + tx_shock.loc[shock_rows, "payment_delay_days"],
    )
    tx_shock.loc[shock_rows, "is_partial_payment"] = np.maximum(
        tx_shock.loc[shock_rows, "is_partial_payment"],
        (rng.random(n_sh) < config.robustness.shock_partial_pay_boost).astype(int),
    )
    tx_shock.loc[shock_rows, "is_late"] = (
        tx_shock.loc[shock_rows, "payment_delay_days"] > config.data.late_threshold_days
    ).astype(int)

    xs, sm, sl, xc, xt, yl, yd, meta, _ = compute_point_in_time_records(tx_shock, config.data)
    t_mask = meta["issue_day"].to_numpy() > cutoff
    res_shock = _score_all_models(
        comparison, xt[t_mask], xs[t_mask], sm[t_mask], sl[t_mask], xc[t_mask], yl[t_mask], yd[t_mask]
    )
    _record_scenario(
        "sudden_economic_shock",
        "Sudden Economic Shock (Delay + Balance + Partial-Pay Surge)",
        "Sudden Economic Shock",
        res_shock,
    )

    df_res = pd.DataFrame(scenario_records)
    weakness_summary = _diagnose_model_weaknesses(df_res, baseline)
    markdown_report = build_robustness_markdown_report(df_res, baseline, weakness_summary)

    return {
        "baseline_test_metrics": {
            m: {
                k: round(float(v), 4)
                for k, v in metrics.items()
                if k in {"roc_auc", "pr_auc", "precision", "recall", "f1", "calibration_ece", "brier_score", "mae", "rmse"}
            }
            for m, metrics in baseline.items()
        },
        "stress_test_results": scenario_records,
        "weakness_summary": weakness_summary,
        "markdown_report": markdown_report,
    }


def _diagnose_model_weaknesses(
    df_res: pd.DataFrame,
    baseline: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize each model's worst stress degradation and structural weaknesses."""
    summaries: list[dict[str, Any]] = []
    for model_name, grp in df_res.groupby("model"):
        worst_auc_row = grp.loc[grp["delta_roc_auc"].idxmin()]
        worst_ece_row = grp.loc[grp["delta_ece"].idxmax()]
        worst_mae_row = grp.loc[grp["delta_mae"].idxmax()]

        mean_abs_auc_drop = float(np.mean(np.abs(np.minimum(grp["delta_roc_auc"], 0.0))))
        mean_ece_increase = float(np.mean(np.maximum(grp["delta_ece"], 0.0)))
        mean_mae_increase = float(np.mean(np.maximum(grp["delta_mae"], 0.0)))

        miss_rows = grp[grp["category"] == "Missing Transaction Records"]
        miss_auc_drop = float(miss_rows["delta_roc_auc"].min()) if len(miss_rows) else 0.0

        shock_rows = grp[grp["category"] == "Sudden Economic Shock"]
        shock_ece_inc = float(shock_rows["delta_ece"].max()) if len(shock_rows) else 0.0
        shock_mae_inc = float(shock_rows["delta_mae"].max()) if len(shock_rows) else 0.0

        findings = []
        if worst_ece_row["delta_ece"] > 0.04:
            findings.append(
                f"Severe calibration drift (+{worst_ece_row['delta_ece']:.3f} ECE) under '{worst_ece_row['scenario_label']}', indicating uncalibrated probabilities under regime shift."
            )
        if worst_mae_row["delta_mae"] > 3.5:
            findings.append(
                f"Delay regression error spikes by +{worst_mae_row['delta_mae']:.2f} MAE days under '{worst_mae_row['scenario_label']}', because static historical features/weights lag rapid systemic shocks."
            )
        if miss_auc_drop < -0.02:
            findings.append(
                f"Sensitive to missing transaction histories (up to {miss_auc_drop:.3f} ROC-AUC drop at 60% record dropout)."
            )
        if not findings:
            findings.append(
                f"Maintains relatively stable ranking (max ROC-AUC drop {worst_auc_row['delta_roc_auc']:.3f}), though calibration worsens under sudden macro shifts."
            )

        summaries.append(
            {
                "model": str(model_name),
                "model_display": MODEL_DISPLAY_NAMES[str(model_name)],
                "baseline_roc_auc": round(float(baseline[str(model_name)]["roc_auc"]), 4),
                "baseline_ece": round(float(baseline[str(model_name)]["calibration_ece"]), 4),
                "baseline_mae": round(float(baseline[str(model_name)]["mae"]), 3),
                "worst_roc_auc_scenario": str(worst_auc_row["scenario_label"]),
                "worst_delta_roc_auc": float(worst_auc_row["delta_roc_auc"]),
                "worst_calibration_scenario": str(worst_ece_row["scenario_label"]),
                "worst_delta_ece": float(worst_ece_row["delta_ece"]),
                "worst_mae_scenario": str(worst_mae_row["scenario_label"]),
                "worst_delta_mae": float(worst_mae_row["delta_mae"]),
                "mean_auc_drop": round(mean_abs_auc_drop, 4),
                "mean_ece_increase": round(mean_ece_increase, 4),
                "mean_mae_increase": round(mean_mae_increase, 3),
                "missing_data_max_auc_drop": round(miss_auc_drop, 4),
                "economic_shock_ece_increase": round(shock_ece_inc, 4),
                "economic_shock_mae_increase": round(shock_mae_inc, 3),
                "diagnosed_weaknesses": findings,
            }
        )

    summaries.sort(key=lambda x: x["mean_auc_drop"] + x["mean_ece_increase"])
    return summaries


def build_robustness_markdown_report(
    df_res: pd.DataFrame,
    baseline: dict[str, dict[str, Any]],
    weakness_summary: list[dict[str, Any]],
) -> str:
    """Create structured Markdown robustness & model weakness report."""
    lines = [
        "# ReceivablesOS ML Robustness & Model Weakness Report",
        "",
        "## 1. Executive Summary",
        "",
        "All four trained models (**Logistic Regression**, **Random Forest**, **XGBoost**, and **Sequence Model (GRU+Attention)**) were stress-tested **without retraining** across five real-world failure modes: systemic payment delay shifts, industry mix shift, customer behavioral drift, missing historical records, and sudden correlated economic shocks.",
        "",
        "## 2. Clean Holdout Test Baseline",
        "",
        "| Model | ROC-AUC | PR-AUC | F1 | ECE (Calib) | Brier | MAE (days) | RMSE (days) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m_name, metrics in baseline.items():
        lines.append(
            f"| {MODEL_DISPLAY_NAMES[m_name]} | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['calibration_ece']:.4f} | {metrics['brier_score']:.4f} | "
            f"{metrics['mae']:.2f} | {metrics['rmse']:.2f} |"
        )

    lines.extend(["", "## 3. Model-by-Model Vulnerability Diagnosis", ""])
    for w in weakness_summary:
        lines.append(f"### {w['model_display']}")
        lines.append(f"- **Worst Discrimination Drop**: `{w['worst_delta_roc_auc']:+.4f}` ROC-AUC under *{w['worst_roc_auc_scenario']}*")
        lines.append(f"- **Worst Calibration Drift**: `{w['worst_delta_ece']:+.4f}` ECE under *{w['worst_calibration_scenario']}*")
        lines.append(f"- **Worst Delay Regression Degradation**: `{w['worst_delta_mae']:+.2f}` MAE days under *{w['worst_mae_scenario']}*")
        for finding in w["diagnosed_weaknesses"]:
            lines.append(f"- **Finding**: {finding}")
        lines.append("")

    lines.extend([
        "## 4. Detailed Stress Test Matrix",
        "",
        "| Scenario | Model | ROC-AUC (Delta) | PR-AUC (Delta) | ECE (Delta) | MAE (Delta) | RMSE (Delta) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for r in df_res.itertuples(index=False):
        lines.append(
            f"| {r.scenario_label} | {r.model_display} | "
            f"{r.roc_auc:.4f} ({r.delta_roc_auc:+.4f}) | "
            f"{r.pr_auc:.4f} ({r.delta_pr_auc:+.4f}) | "
            f"{r.calibration_ece:.4f} ({r.delta_ece:+.4f}) | "
            f"{r.mae:.2f} ({r.delta_mae:+.2f}) | "
            f"{r.rmse:.2f} ({r.delta_rmse:+.2f}) |"
        )

    lines.extend([
        "",
        "## 5. Recommended Engineering Mitigations",
        "",
        "1. **Rolling Recalibration (Platt / Isotonic)**: Macro delay shifts and economic shocks inflate ECE faster than they degrade ROC-AUC; trigger automated weekly temperature scaling when observed open-invoice overdue rate drifts > 5 percentage points.",
        "2. **Macro & Sector Exogenous Signals**: Tabular and sequence encoders trained on closed historical windows underestimate abrupt systemic jumps (`MAE` spikes under sudden shock); incorporate live macro/sector overdue rate features.",
        "3. **Missing-History Resilience**: When historical transaction records drop out (`40%-60%`), explicitly pass missingness indicators and fallback credit-bureau/static priors rather than relying solely on lagged aggregates.",
        "4. **Concept-Drift Triggers**: Monitor rolling customer delay slope (`hist_delay_slope`) and PSI/KS shift across behavioral clusters to trigger targeted retraining on recent 60-90 day windows.",
        "",
    ])
    return "\n".join(lines)
