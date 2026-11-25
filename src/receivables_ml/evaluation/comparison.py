"""Unified model training, 8-metric comparison, and Sequence vs XGBoost analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from receivables_ml.config import ExperimentConfig
from receivables_ml.data.features import PreparedDatasets
from receivables_ml.evaluation.metrics import (
    ModelPrediction,
    evaluate_predictions,
    find_optimal_f1_threshold,
)
from receivables_ml.models.sequence import SequenceRiskModel, analyze_temporal_embeddings
from receivables_ml.models.tabular import (
    BaseDualTabularModel,
    XGBoostSuite,
    build_tabular_models,
)

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "sequence_model": "Sequence Model (GRU+Attn)",
}


@dataclass
class ComparisonArtifacts:
    """Trained models, split metrics, predictions, curves, and embedding analysis."""

    tabular_models: dict[str, BaseDualTabularModel]
    sequence_model: SequenceRiskModel
    thresholds: dict[str, float]
    val_metrics: dict[str, dict[str, Any]]
    test_metrics: dict[str, dict[str, Any]]
    test_predictions: dict[str, ModelPrediction]
    curve_data: dict[str, dict[str, list[float]]]
    sequence_vs_xgboost: dict[str, Any]
    embedding_analysis: dict[str, Any]
    hybrid_ablation: dict[str, Any]
    train_embeddings: np.ndarray
    test_embeddings: np.ndarray
    test_attention: np.ndarray


def _downsample_curve(x: np.ndarray, y: np.ndarray, max_points: int = 80) -> tuple[list[float], list[float]]:
    if len(x) <= max_points:
        return [round(float(v), 4) for v in x], [round(float(v), 4) for v in y]
    idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
    return [round(float(x[i]), 4) for i in idx], [round(float(y[i]), 4) for i in idx]


def run_model_comparison(
    datasets: PreparedDatasets,
    config: ExperimentConfig,
) -> ComparisonArtifacts:
    """Train Logistic Regression, Random Forest, XGBoost, and Sequence Model and compare."""
    tr, va, te = datasets.train, datasets.val, datasets.test
    seed = config.seed

    # 1. Train 3 tabular models
    tabular_models = build_tabular_models(config.tabular, seed=seed)
    val_preds: dict[str, ModelPrediction] = {}
    test_preds: dict[str, ModelPrediction] = {}
    thresholds: dict[str, float] = {}
    val_metrics: dict[str, dict[str, Any]] = {}
    test_metrics: dict[str, dict[str, Any]] = {}

    for name, model in tabular_models.items():
        model.fit(tr.X_tab, tr.y_late, tr.y_delay, va.X_tab, va.y_late, va.y_delay)
        vp = model.predict(va.X_tab)
        tp = model.predict(te.X_tab)
        thr = find_optimal_f1_threshold(va.y_late, vp.probs)

        val_preds[name] = vp
        test_preds[name] = tp
        thresholds[name] = thr
        val_metrics[name] = evaluate_predictions(
            va.y_late, va.y_delay, vp.probs, vp.delay_preds, threshold=thr, n_bootstrap=0, seed=seed
        )
        test_metrics[name] = evaluate_predictions(
            te.y_late, te.y_delay, tp.probs, tp.delay_preds, threshold=thr, n_bootstrap=200, seed=seed
        )

    # 2. Train PyTorch Temporal Sequence Model (GRU + Attention)
    seq_model = SequenceRiskModel(
        seq_input_dim=tr.X_seq.shape[-1],
        ctx_input_dim=tr.X_ctx.shape[-1],
        config=config.sequence,
        seed=seed,
    )
    seq_model.fit(
        tr.X_seq, tr.seq_mask, tr.seq_lengths, tr.X_ctx, tr.y_late, tr.y_delay,
        va.X_seq, va.seq_mask, va.seq_lengths, va.X_ctx, va.y_late, va.y_delay,
    )
    seq_vp = seq_model.predict(va.X_seq, va.seq_mask, va.seq_lengths, va.X_ctx)
    seq_tp = seq_model.predict(te.X_seq, te.seq_mask, te.seq_lengths, te.X_ctx)
    seq_thr = find_optimal_f1_threshold(va.y_late, seq_vp.probs)

    val_preds["sequence_model"] = seq_vp
    test_preds["sequence_model"] = seq_tp
    thresholds["sequence_model"] = seq_thr
    val_metrics["sequence_model"] = evaluate_predictions(
        va.y_late, va.y_delay, seq_vp.probs, seq_vp.delay_preds, threshold=seq_thr, n_bootstrap=0, seed=seed
    )
    test_metrics["sequence_model"] = evaluate_predictions(
        te.y_late, te.y_delay, seq_tp.probs, seq_tp.delay_preds, threshold=seq_thr, n_bootstrap=200, seed=seed
    )

    # 3. Extract Temporal Customer Embeddings & Attention Weights
    tr_emb, _ = seq_model.extract_embeddings(tr.X_seq, tr.seq_mask, tr.seq_lengths)
    va_emb, _ = seq_model.extract_embeddings(va.X_seq, va.seq_mask, va.seq_lengths)
    te_emb, te_attn = seq_model.extract_embeddings(te.X_seq, te.seq_mask, te.seq_lengths)

    emb_analysis = analyze_temporal_embeddings(
        train_embeddings=tr_emb,
        train_meta=tr.meta,
        test_embeddings=te_emb,
        test_attn=te_attn,
        test_meta=te.meta,
        seed=seed,
    )

    # 4. Hybrid Ablation: XGBoost + Temporal Customer Embeddings
    xgb_hybrid = XGBoostSuite(config.tabular, seed=seed)
    xgb_hybrid.fit(
        np.hstack([tr.X_tab, tr_emb]), tr.y_late, tr.y_delay,
        np.hstack([va.X_tab, va_emb]), va.y_late, va.y_delay,
    )
    hyb_vp = xgb_hybrid.predict(np.hstack([va.X_tab, va_emb]))
    hyb_tp = xgb_hybrid.predict(np.hstack([te.X_tab, te_emb]))
    hyb_thr = find_optimal_f1_threshold(va.y_late, hyb_vp.probs)
    hyb_metrics = evaluate_predictions(
        te.y_late, te.y_delay, hyb_tp.probs, hyb_tp.delay_preds, threshold=hyb_thr, n_bootstrap=0, seed=seed
    )
    hybrid_ablation = {
        "xgboost_tabular_only": {
            "roc_auc": test_metrics["xgboost"]["roc_auc"],
            "pr_auc": test_metrics["xgboost"]["pr_auc"],
            "calibration_ece": test_metrics["xgboost"]["calibration_ece"],
            "mae": test_metrics["xgboost"]["mae"],
            "rmse": test_metrics["xgboost"]["rmse"],
        },
        "xgboost_plus_temporal_embeddings": {
            "roc_auc": hyb_metrics["roc_auc"],
            "pr_auc": hyb_metrics["pr_auc"],
            "calibration_ece": hyb_metrics["calibration_ece"],
            "mae": hyb_metrics["mae"],
            "rmse": hyb_metrics["rmse"],
        },
        "delta_roc_auc": round(hyb_metrics["roc_auc"] - test_metrics["xgboost"]["roc_auc"], 5),
        "delta_pr_auc": round(hyb_metrics["pr_auc"] - test_metrics["xgboost"]["pr_auc"], 5),
        "delta_mae": round(hyb_metrics["mae"] - test_metrics["xgboost"]["mae"], 4),
    }

    # 5. ROC & PR Curve Data for Dashboard
    curve_data: dict[str, dict[str, list[float]]] = {}
    for name, pred in test_preds.items():
        fpr, tpr, _ = roc_curve(te.y_late, pred.probs)
        prec, rec, _ = precision_recall_curve(te.y_late, pred.probs)
        fpr_ds, tpr_ds = _downsample_curve(fpr, tpr)
        rec_ds, prec_ds = _downsample_curve(rec[::-1], prec[::-1])
        curve_data[name] = {
            "fpr": fpr_ds,
            "tpr": tpr_ds,
            "recall": rec_ds,
            "precision": prec_ds,
        }

    # 6. Sequence vs XGBoost Deep Slice Comparison
    seq_vs_xgb = _compare_sequence_vs_xgboost(
        te=te,
        xgb_pred=test_preds["xgboost"],
        seq_pred=test_preds["sequence_model"],
        xgb_thr=thresholds["xgboost"],
        seq_thr=thresholds["sequence_model"],
        xgb_metrics=test_metrics["xgboost"],
        seq_metrics=test_metrics["sequence_model"],
    )

    return ComparisonArtifacts(
        tabular_models=tabular_models,
        sequence_model=seq_model,
        thresholds=thresholds,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        test_predictions=test_preds,
        curve_data=curve_data,
        sequence_vs_xgboost=seq_vs_xgb,
        embedding_analysis=emb_analysis,
        hybrid_ablation=hybrid_ablation,
        train_embeddings=tr_emb,
        test_embeddings=te_emb,
        test_attention=te_attn,
    )


def _compare_sequence_vs_xgboost(
    te: Any,
    xgb_pred: ModelPrediction,
    seq_pred: ModelPrediction,
    xgb_thr: float,
    seq_thr: float,
    xgb_metrics: dict[str, Any],
    seq_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Slice-level comparison between Sequence Model and XGBoost."""
    overall_delta = {
        "roc_auc_diff_seq_minus_xgb": round(seq_metrics["roc_auc"] - xgb_metrics["roc_auc"], 5),
        "pr_auc_diff_seq_minus_xgb": round(seq_metrics["pr_auc"] - xgb_metrics["pr_auc"], 5),
        "f1_diff_seq_minus_xgb": round(seq_metrics["f1"] - xgb_metrics["f1"], 5),
        "ece_diff_seq_minus_xgb": round(seq_metrics["calibration_ece"] - xgb_metrics["calibration_ece"], 5),
        "mae_diff_seq_minus_xgb": round(seq_metrics["mae"] - xgb_metrics["mae"], 4),
        "rmse_diff_seq_minus_xgb": round(seq_metrics["rmse"] - xgb_metrics["rmse"], 4),
    }

    by_archetype: list[dict[str, Any]] = []
    archs = te.meta["true_archetype"].to_numpy()
    for arch in sorted(np.unique(archs)):
        m = archs == arch
        if len(np.unique(te.y_late[m])) < 2:
            continue
        x_res = evaluate_predictions(te.y_late[m], te.y_delay[m], xgb_pred.probs[m], xgb_pred.delay_preds[m], threshold=xgb_thr, n_bootstrap=0)
        s_res = evaluate_predictions(te.y_late[m], te.y_delay[m], seq_pred.probs[m], seq_pred.delay_preds[m], threshold=seq_thr, n_bootstrap=0)
        by_archetype.append(
            {
                "archetype": str(arch),
                "n_invoices": int(np.sum(m)),
                "late_rate": round(float(np.mean(te.y_late[m])), 3),
                "xgb_roc_auc": round(x_res["roc_auc"], 4),
                "seq_roc_auc": round(s_res["roc_auc"], 4),
                "xgb_mae": round(x_res["mae"], 2),
                "seq_mae": round(s_res["mae"], 2),
            }
        )

    by_history_len: list[dict[str, Any]] = []
    buckets = [
        ("short_3_to_6", (te.seq_lengths >= 3) & (te.seq_lengths <= 6)),
        ("medium_7_to_12", (te.seq_lengths >= 7) & (te.seq_lengths <= 12)),
        ("long_13_plus", te.seq_lengths >= 13),
    ]
    for label, m in buckets:
        if np.sum(m) < 20 or len(np.unique(te.y_late[m])) < 2:
            continue
        x_res = evaluate_predictions(te.y_late[m], te.y_delay[m], xgb_pred.probs[m], xgb_pred.delay_preds[m], threshold=xgb_thr, n_bootstrap=0)
        s_res = evaluate_predictions(te.y_late[m], te.y_delay[m], seq_pred.probs[m], seq_pred.delay_preds[m], threshold=seq_thr, n_bootstrap=0)
        by_history_len.append(
            {
                "history_bucket": label,
                "n_invoices": int(np.sum(m)),
                "xgb_roc_auc": round(x_res["roc_auc"], 4),
                "seq_roc_auc": round(s_res["roc_auc"], 4),
                "xgb_mae": round(x_res["mae"], 2),
                "seq_mae": round(s_res["mae"], 2),
            }
        )

    return {
        "overall_delta": overall_delta,
        "by_archetype": by_archetype,
        "by_history_length": by_history_len,
    }
