"""Comprehensive classification, calibration, and delay regression metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


@dataclass
class ModelPrediction:
    """Unified prediction container for both risk classification and delay regression."""

    probs: np.ndarray
    delay_preds: np.ndarray


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute Brier score, Expected Calibration Error (ECE), MCE, and bin statistics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)

    brier = float(brier_score_loss(y_true, y_prob))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0
    bins_detail: list[dict[str, float | int]] = []
    n_total = len(y_true)

    for idx in range(n_bins):
        lo, hi = float(bin_edges[idx]), float(bin_edges[idx + 1])
        if idx == n_bins - 1:
            in_bin = (y_prob >= lo) & (y_prob <= hi)
        else:
            in_bin = (y_prob >= lo) & (y_prob < hi)

        count = int(np.sum(in_bin))
        if count > 0:
            mean_conf = float(np.mean(y_prob[in_bin]))
            emp_rate = float(np.mean(y_true[in_bin]))
            gap = abs(mean_conf - emp_rate)
            ece += (count / max(n_total, 1)) * gap
            mce = max(mce, gap)
        else:
            mean_conf = 0.5 * (lo + hi)
            emp_rate = float("nan")

        bins_detail.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "bin_center": 0.5 * (lo + hi),
                "count": count,
                "mean_pred_prob": mean_conf,
                "empirical_pos_rate": emp_rate if not np.isnan(emp_rate) else None,
            }
        )

    return {
        "brier_score": brier,
        "ece": float(ece),
        "mce": float(mce),
        "bins": bins_detail,
    }


def find_optimal_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid_size: int = 91,
) -> float:
    """Select classification threshold maximizing F1 on validation probabilities."""
    thresholds = np.linspace(0.05, 0.95, grid_size)
    best_thresh = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        score = float(f1_score(y_true, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thr)
    return best_thresh


def evaluate_predictions(
    y_late: np.ndarray,
    y_delay: np.ndarray,
    probs: np.ndarray,
    delay_preds: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute all 8 requested metrics plus calibration bins and bootstrap CIs."""
    y_late = np.asarray(y_late, dtype=np.int64)
    y_delay = np.asarray(y_delay, dtype=np.float64)
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    delay_preds = np.asarray(delay_preds, dtype=np.float64)

    preds_bin = (probs >= threshold).astype(np.int64)
    has_both_classes = len(np.unique(y_late)) == 2

    roc_auc = float(roc_auc_score(y_late, probs)) if has_both_classes else float("nan")
    pr_auc = float(average_precision_score(y_late, probs)) if has_both_classes else float("nan")
    precision = float(precision_score(y_late, preds_bin, zero_division=0))
    recall = float(recall_score(y_late, preds_bin, zero_division=0))
    f1 = float(f1_score(y_late, preds_bin, zero_division=0))
    accuracy = float(accuracy_score(y_late, preds_bin))

    cal = compute_calibration_metrics(y_late, probs, n_bins=10)
    mae = float(mean_absolute_error(y_delay, delay_preds))
    rmse = float(root_mean_squared_error(y_delay, delay_preds))

    ci: dict[str, list[float]] = {}
    if n_bootstrap > 0 and len(y_late) >= 20 and has_both_classes:
        rng = np.random.default_rng(seed)
        n = len(y_late)
        aucs, prs, maes = [], [], []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            yb = y_late[idx]
            if len(np.unique(yb)) < 2:
                continue
            aucs.append(float(roc_auc_score(yb, probs[idx])))
            prs.append(float(average_precision_score(yb, probs[idx])))
            maes.append(float(mean_absolute_error(y_delay[idx], delay_preds[idx])))
        if aucs:
            ci["roc_auc_95ci"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
            ci["pr_auc_95ci"] = [float(np.percentile(prs, 2.5)), float(np.percentile(prs, 97.5))]
            ci["mae_95ci"] = [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))]

    return {
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "calibration_ece": cal["ece"],
        "calibration_mce": cal["mce"],
        "brier_score": cal["brier_score"],
        "mae": mae,
        "rmse": rmse,
        "positive_rate": float(np.mean(y_late)),
        "predicted_positive_rate": float(np.mean(preds_bin)),
        "calibration_bins": cal["bins"],
        "bootstrap_ci": ci,
    }
