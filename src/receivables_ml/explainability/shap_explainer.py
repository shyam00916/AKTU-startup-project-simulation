"""SHAP explainability and temporal risk driver attribution."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
import torch

from receivables_ml.config import ExplainabilityConfig
from receivables_ml.data.features import SequenceDatasetArrays
from receivables_ml.models.sequence import SequenceRiskModel
from receivables_ml.models.tabular import LogisticRegressionSuite, RandomForestSuite, XGBoostSuite

BALANCE_FEATURES: set[str] = {
    "outstanding_balance_before_log",
    "balance_utilization",
    "balance_plus_new_utilization",
    "amount_to_limit_ratio",
    "open_invoices_before",
    "overdue_open_balance_ratio",
    "hist_balance_util_mean",
    "hist_balance_util_last",
    "hist_balance_util_slope",
    "hist_unpaid_ratio_mean",
}

DELAY_HISTORY_FEATURES: set[str] = {
    "hist_delay_mean",
    "hist_delay_std",
    "hist_delay_max",
    "hist_delay_last",
    "hist_delay_ewm",
    "hist_delay_slope",
    "hist_late_rate",
    "hist_early_rate",
}


def _extract_shap_matrix(explainer: Any, X: np.ndarray) -> np.ndarray:
    """Normalize SHAP output across SHAP versions and binary/single output shapes."""
    vals = explainer.shap_values(X)
    if isinstance(vals, list):
        arr = np.asarray(vals[1] if len(vals) > 1 else vals[0], dtype=np.float64)
    else:
        arr = np.asarray(vals, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


def _compute_sequence_step_saliency(
    seq_model: SequenceRiskModel,
    X_seq: np.ndarray,
    seq_mask: np.ndarray,
    seq_lengths: np.ndarray,
    X_ctx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return attention weights and gradient-input step importance for sequence rows."""
    seq_model.net.eval()
    seq_model.net.zero_grad(set_to_none=True)
    X_ctx_scaled = seq_model.ctx_scaler.transform(X_ctx).astype(np.float32)

    with torch.enable_grad():
        t_seq = torch.tensor(X_seq, dtype=torch.float32, requires_grad=True)
        t_mask = torch.tensor(seq_mask, dtype=torch.float32)
        t_len = torch.tensor(seq_lengths, dtype=torch.long)
        t_ctx = torch.tensor(X_ctx_scaled, dtype=torch.float32)

        logits, _, _, attn = seq_model.net(t_seq, t_mask, t_len, t_ctx)
        logits.sum().backward()

        grad_seq = t_seq.grad.detach().numpy()
        saliency = np.sum(np.abs(grad_seq * X_seq), axis=-1) * seq_mask
    seq_model.net.zero_grad(set_to_none=True)
    return attn.detach().numpy(), saliency


def generate_shap_explanations(
    xgb_model: XGBoostSuite,
    rf_model: RandomForestSuite,
    logreg_model: LogisticRegressionSuite,
    seq_model: SequenceRiskModel,
    train_ds: SequenceDatasetArrays,
    test_ds: SequenceDatasetArrays,
    tab_feature_names: list[str],
    xgb_test_probs: np.ndarray,
    seq_test_probs: np.ndarray,
    config: ExplainabilityConfig,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute global SHAP importances and per-customer explanations for every high-risk customer."""
    rng = np.random.default_rng(seed)
    n_bg = min(config.shap_background_samples, len(train_ds.X_tab))
    bg_idx = rng.choice(len(train_ds.X_tab), size=n_bg, replace=False)
    X_bg = train_ds.X_tab[bg_idx]

    # 1. XGBoost TreeExplainer on full test split
    xgb_explainer = shap.TreeExplainer(xgb_model.classifier)
    xgb_shap = _extract_shap_matrix(xgb_explainer, test_ds.X_tab)
    base_val = xgb_explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(np.asarray(base_val).ravel()[-1])
    else:
        base_val = float(base_val)

    # 2. Logistic Regression LinearExplainer on standardized pipeline features
    lr_imputer = logreg_model.classifier.named_steps["imputer"]
    lr_scaler = logreg_model.classifier.named_steps["scaler"]
    lr_clf = logreg_model.classifier.named_steps["clf"]
    X_bg_lr = lr_scaler.transform(lr_imputer.transform(X_bg))
    X_te_lr = lr_scaler.transform(lr_imputer.transform(test_ds.X_tab))
    lr_masker = shap.maskers.Independent(X_bg_lr, max_samples=len(X_bg_lr))
    lr_explainer = shap.LinearExplainer(lr_clf, lr_masker)
    lr_shap = _extract_shap_matrix(lr_explainer, X_te_lr)

    # 3. Random Forest TreeExplainer on sample for cross-model global agreement
    n_rf_sample = min(250, len(test_ds.X_tab))
    rf_X_imp = rf_model.imputer.transform(test_ds.X_tab[:n_rf_sample])
    rf_explainer = shap.TreeExplainer(rf_model.classifier)
    rf_shap = _extract_shap_matrix(rf_explainer, rf_X_imp)

    # Global SHAP summary table
    xgb_mean_abs = np.mean(np.abs(xgb_shap), axis=0)
    lr_mean_abs = np.mean(np.abs(lr_shap), axis=0)
    rf_mean_abs = np.mean(np.abs(rf_shap), axis=0)

    global_importance = []
    for j, fname in enumerate(tab_feature_names):
        group = (
            "Outstanding Balance & Utilization"
            if fname in BALANCE_FEATURES
            else "Previous Payment Delays"
            if fname in DELAY_HISTORY_FEATURES
            else "Temporal & Invoice Context"
        )
        global_importance.append(
            {
                "feature": fname,
                "feature_group": group,
                "xgb_mean_abs_shap": round(float(xgb_mean_abs[j]), 5),
                "rf_mean_abs_shap": round(float(rf_mean_abs[j]), 5),
                "logreg_mean_abs_shap": round(float(lr_mean_abs[j]), 5),
            }
        )
    global_importance.sort(key=lambda r: r["xgb_mean_abs_shap"], reverse=True)

    # Group-level contribution totals
    bal_indices = [i for i, f in enumerate(tab_feature_names) if f in BALANCE_FEATURES]
    delay_indices = [i for i, f in enumerate(tab_feature_names) if f in DELAY_HISTORY_FEATURES]

    # Identify EVERY high-risk customer in test split (latest test invoice per customer with risk >= threshold)
    test_meta = test_ds.meta.copy()
    test_meta["row_idx"] = np.arange(len(test_meta))
    test_meta["xgb_prob"] = xgb_test_probs
    test_meta["seq_prob"] = seq_test_probs
    test_meta["combined_risk_prob"] = 0.5 * (xgb_test_probs + seq_test_probs)

    latest_per_cust = (
        test_meta.sort_values(["customer_id", "seq_idx"])
        .groupby("customer_id", as_index=False)
        .tail(1)
        .sort_values("xgb_prob", ascending=False)
    )
    high_risk_df = latest_per_cust[latest_per_cust["xgb_prob"] >= config.high_risk_threshold]
    if len(high_risk_df) == 0:
        high_risk_df = latest_per_cust.head(min(25, len(latest_per_cust)))
    high_risk_df = high_risk_df.head(config.max_high_risk_customers_export)

    hr_indices = high_risk_df["row_idx"].to_numpy(dtype=int)
    attn_hr, saliency_hr = _compute_sequence_step_saliency(
        seq_model=seq_model,
        X_seq=test_ds.X_seq[hr_indices],
        seq_mask=test_ds.seq_mask[hr_indices],
        seq_lengths=test_ds.seq_lengths[hr_indices],
        X_ctx=test_ds.X_ctx[hr_indices],
    )

    high_risk_explanations: list[dict[str, Any]] = []
    for pos, rec in enumerate(high_risk_df.itertuples(index=False)):
        r_idx = int(rec.row_idx)
        shap_row = xgb_shap[r_idx]
        x_row = test_ds.X_tab[r_idx]

        # Sort features by |SHAP|
        order = np.argsort(-np.abs(shap_row))
        top_factors = []
        for j in order[:8]:
            fname = tab_feature_names[int(j)]
            top_factors.append(
                {
                    "feature": fname,
                    "feature_value": round(float(x_row[int(j)]), 4),
                    "shap_contribution": round(float(shap_row[int(j)]), 5),
                    "direction": "increases_risk" if shap_row[int(j)] >= 0 else "reduces_risk",
                }
            )

        bal_breakdown = {
            tab_feature_names[j]: round(float(shap_row[j]), 5)
            for j in bal_indices
            if abs(shap_row[j]) > 1e-4
        }
        delay_breakdown = {
            tab_feature_names[j]: round(float(shap_row[j]), 5)
            for j in delay_indices
            if abs(shap_row[j]) > 1e-4
        }

        bal_total = float(np.sum(shap_row[bal_indices]))
        delay_total = float(np.sum(shap_row[delay_indices]))
        other_total = float(np.sum(shap_row) - bal_total - delay_total)

        seq_len = int(test_ds.seq_lengths[r_idx])
        attn_valid = [round(float(v), 4) for v in attn_hr[pos, :seq_len]]
        sal_valid = [round(float(v), 4) for v in saliency_hr[pos, :seq_len]]

        high_risk_explanations.append(
            {
                "customer_id": rec.customer_id,
                "invoice_id": rec.invoice_id,
                "industry": rec.industry,
                "true_archetype": rec.true_archetype,
                "xgb_risk_prob": round(float(rec.xgb_prob), 4),
                "sequence_risk_prob": round(float(rec.seq_prob), 4),
                "actual_is_late": int(test_ds.y_late[r_idx]),
                "actual_delay_days": float(test_ds.y_delay[r_idx]),
                "shap_base_log_odds": round(base_val, 4),
                "most_influential_risk_factors": top_factors,
                "historical_payment_trends": {
                    "recent_delays_days": rec.recent_delays,
                    "recent_outstanding_balances": rec.recent_balances,
                    "hist_delay_mean_days": round(float(rec.hist_delay_mean), 2),
                    "hist_delay_last_days": round(float(rec.hist_delay_last), 2),
                    "hist_delay_slope": round(float(rec.hist_delay_slope), 3),
                    "hist_partial_payment_rate": round(float(rec.hist_partial_rate), 3),
                    "hist_tx_interval_mean_days": round(float(rec.hist_tx_interval_mean), 2),
                    "sequence_attention_weights": attn_valid,
                    "sequence_step_saliency": sal_valid,
                },
                "outstanding_balance_contribution": {
                    "total_shap_log_odds": round(bal_total, 5),
                    "outstanding_balance_before": round(float(rec.outstanding_balance_before), 2),
                    "balance_utilization": round(float(rec.balance_utilization), 3),
                    "feature_breakdown": bal_breakdown,
                },
                "previous_payment_delays_contribution": {
                    "total_shap_log_odds": round(delay_total, 5),
                    "feature_breakdown": delay_breakdown,
                },
                "other_context_contribution_log_odds": round(other_total, 5),
            }
        )

    return {
        "n_test_scored": int(len(test_ds.X_tab)),
        "high_risk_threshold": float(config.high_risk_threshold),
        "n_high_risk_customers_identified": int(len(latest_per_cust[latest_per_cust["xgb_prob"] >= config.high_risk_threshold])),
        "n_high_risk_customers_explained": int(len(high_risk_explanations)),
        "global_feature_importance": global_importance,
        "high_risk_customer_explanations": high_risk_explanations,
    }
