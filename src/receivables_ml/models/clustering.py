"""Customer payment behavior clustering (K-Means + DBSCAN) and segmentation impact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from receivables_ml.config import ClusteringConfig, TabularModelsConfig
from receivables_ml.data.features import PreparedDatasets
from receivables_ml.evaluation.metrics import evaluate_predictions, find_optimal_f1_threshold
from receivables_ml.models.tabular import LogisticRegressionSuite, XGBoostSuite

BEHAVIOR_CLUSTER_FEATURES: tuple[str, ...] = (
    "delay_mean",
    "delay_std",
    "delay_slope",
    "early_pay_rate",
    "late_rate",
    "max_delay",
    "tx_frequency_per_month",
    "inter_invoice_cv",
    "log_amount_mean",
    "log_amount_std",
    "partial_payment_rate",
    "avg_installments",
    "avg_balance_utilization",
    "balance_utilization_slope",
)


PERSONA_DESCRIPTIONS: dict[str, str] = {
    "Consistent Early Payers": "Pays early or on-time with minimal delay variance and rare partial installments.",
    "Consistent Late Payers": "Persistently delayed past terms with moderate predictability and steady cadence.",
    "Irregular Payers": "High payment-delay volatility, erratic inter-invoice cadence, and bimodal timing spikes.",
    "High-Value Delayed Payers": "Large invoice tickets, elevated utilization, high partial-payment rate, and extended delays.",
    "Deteriorating Customers": "Positive delay and balance-utilization slopes indicating worsening liquidity trajectory.",
}


def _assign_cluster_personas(centroids_df: pd.DataFrame) -> dict[int, str]:
    """Map unsupervised cluster centroids to the 5 business behavior personas via Hungarian matching."""
    personas = list(PERSONA_DESCRIPTIONS.keys())
    k = len(centroids_df)
    if k != len(personas):
        return {int(cid): f"Behavior Cluster {cid}" for cid in centroids_df.index}

    # Z-score across cluster centroids so dimensions are comparable
    z = (centroids_df - centroids_df.mean()) / (centroids_df.std() + 1e-6)
    cost = np.zeros((k, k), dtype=np.float64)

    for i in range(k):
        row = z.iloc[i]
        # Negative affinity = cost for Hungarian minimization
        affinities = [
            # Consistent Early Payers: high early_pay_rate, low delay_mean & delay_std
            1.6 * row["early_pay_rate"] - 1.4 * row["delay_mean"] - 0.8 * row["delay_std"],
            # Consistent Late Payers: high delay_mean, moderate-low delay_std, low delay_slope, moderate amount
            1.4 * row["delay_mean"] - 0.9 * row["delay_std"] - 0.7 * row["delay_slope"] - 0.5 * row["log_amount_mean"],
            # Irregular Payers: high delay_std & inter_invoice_cv
            1.6 * row["delay_std"] + 1.4 * row["inter_invoice_cv"] + 0.5 * row["log_amount_std"],
            # High-Value Delayed Payers: high log_amount_mean, partial_payment_rate, delay_mean
            1.8 * row["log_amount_mean"] + 1.0 * row["partial_payment_rate"] + 0.6 * row["delay_mean"],
            # Deteriorating Customers: high delay_slope & balance_utilization_slope
            1.8 * row["delay_slope"] + 1.3 * row["balance_utilization_slope"] + 0.4 * row["late_rate"],
        ]
        cost[i, :] = -np.asarray(affinities)

    row_ind, col_ind = linear_sum_assignment(cost)
    return {int(centroids_df.index[r]): personas[c] for r, c in zip(row_ind, col_ind)}


@dataclass
class CustomerClusteringArtifacts:
    """Fitted clustering models, mappings, diagnostics, and customer assignments."""

    scaler: StandardScaler
    kmeans: KMeans
    dbscan: DBSCAN
    dbscan_eps: float
    persona_map: dict[int, str]
    train_customer_assignments: pd.DataFrame
    kmeans_k_sweep: list[dict[str, float]]
    dbscan_eps_sweep: list[dict[str, float]]
    cluster_profiles: list[dict[str, Any]]
    dbscan_summary: dict[str, Any]
    alignment_metrics: dict[str, float]
    segmentation_lift: dict[str, Any]


def fit_clustering_and_segmentation(
    datasets: PreparedDatasets,
    clustering_cfg: ClusteringConfig,
    tabular_cfg: TabularModelsConfig,
    seed: int = 42,
) -> CustomerClusteringArtifacts:
    """Run K-Means and DBSCAN on training customer histories and quantify segmentation lift."""
    beh_train = datasets.customer_behavior_train.copy().reset_index(drop=True)
    feat_cols = list(BEHAVIOR_CLUSTER_FEATURES)
    X_beh = beh_train[feat_cols].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_beh)

    # 1. K-Means k-sweep
    k_sweep: list[dict[str, float]] = []
    for k in range(clustering_cfg.kmeans_k_min, clustering_cfg.kmeans_k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed)
        labels = km.fit_predict(X_scaled)
        k_sweep.append(
            {
                "k": float(k),
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(X_scaled, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
                "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
            }
        )

    target_k = clustering_cfg.kmeans_target_k
    kmeans = KMeans(n_clusters=target_k, n_init=30, random_state=seed)
    km_labels = kmeans.fit_predict(X_scaled)
    beh_train["kmeans_cluster"] = km_labels

    centroids_df = beh_train.groupby("kmeans_cluster")[feat_cols].mean()
    persona_map = _assign_cluster_personas(centroids_df)
    beh_train["behavior_segment"] = beh_train["kmeans_cluster"].map(persona_map)

    # 2. DBSCAN eps grid search
    eps_sweep: list[dict[str, float]] = []
    best_eps = clustering_cfg.dbscan_eps_grid[0]
    best_score = -999.0
    best_dbscan = None
    best_db_labels = None

    for eps in clustering_cfg.dbscan_eps_grid:
        db = DBSCAN(eps=float(eps), min_samples=clustering_cfg.dbscan_min_samples)
        db_labels = db.fit_predict(X_scaled)
        n_clusters = len(set(db_labels) - {-1})
        noise_ratio = float(np.mean(db_labels == -1))
        non_noise = db_labels != -1
        if n_clusters >= 2 and np.sum(non_noise) >= 20:
            sil = float(silhouette_score(X_scaled[non_noise], db_labels[non_noise]))
        else:
            sil = -1.0

        eps_sweep.append(
            {
                "eps": float(eps),
                "n_clusters": float(n_clusters),
                "noise_ratio": noise_ratio,
                "silhouette_non_noise": sil,
            }
        )
        # Prefer clean multi-cluster discovery with 5%-35% anomaly isolation
        objective = sil + (0.45 if n_clusters >= 2 else -1.0) - 0.8 * max(0.0, noise_ratio - 0.35)
        if objective > best_score:
            best_score = objective
            best_eps = float(eps)
            best_dbscan = db
            best_db_labels = db_labels

    if best_dbscan is None or best_db_labels is None:
        best_dbscan = DBSCAN(eps=best_eps, min_samples=clustering_cfg.dbscan_min_samples)
        best_db_labels = best_dbscan.fit_predict(X_scaled)

    beh_train["dbscan_cluster"] = best_db_labels
    beh_train["is_dbscan_outlier"] = (best_db_labels == -1).astype(int)

    # Build cluster profiles
    profiles: list[dict[str, Any]] = []
    for cid, grp in beh_train.groupby("kmeans_cluster"):
        persona = persona_map[int(cid)]
        arch_dist = grp["true_archetype"].value_counts(normalize=True).to_dict()
        profiles.append(
            {
                "cluster_id": int(cid),
                "segment_name": persona,
                "description": PERSONA_DESCRIPTIONS.get(persona, ""),
                "n_customers": int(len(grp)),
                "share_pct": round(100.0 * len(grp) / len(beh_train), 2),
                "mean_delay_days": round(float(grp["delay_mean"].mean()), 2),
                "std_delay_days": round(float(grp["delay_std"].mean()), 2),
                "delay_slope": round(float(grp["delay_slope"].mean()), 3),
                "late_rate": round(float(grp["late_rate"].mean()), 3),
                "early_pay_rate": round(float(grp["early_pay_rate"].mean()), 3),
                "mean_invoice_amount": round(float(np.exp(grp["log_amount_mean"].mean())), 2),
                "partial_payment_rate": round(float(grp["partial_payment_rate"].mean()), 3),
                "avg_balance_utilization": round(float(grp["avg_balance_utilization"].mean()), 3),
                "tx_frequency_per_month": round(float(grp["tx_frequency_per_month"].mean()), 2),
                "dbscan_outlier_rate": round(float(grp["is_dbscan_outlier"].mean()), 3),
                "dominant_true_archetype": max(arch_dist, key=arch_dist.get),
            }
        )

    outlier_grp = beh_train[beh_train["is_dbscan_outlier"] == 1]
    inlier_grp = beh_train[beh_train["is_dbscan_outlier"] == 0]
    dbscan_summary = {
        "selected_eps": best_eps,
        "min_samples": clustering_cfg.dbscan_min_samples,
        "n_core_clusters": int(len(set(best_db_labels) - {-1})),
        "n_outliers": int(len(outlier_grp)),
        "outlier_pct": round(100.0 * len(outlier_grp) / len(beh_train), 2),
        "outlier_mean_late_rate": round(float(outlier_grp["late_rate"].mean()), 3) if len(outlier_grp) else 0.0,
        "inlier_mean_late_rate": round(float(inlier_grp["late_rate"].mean()), 3) if len(inlier_grp) else 0.0,
        "outlier_mean_delay_std": round(float(outlier_grp["delay_std"].mean()), 2) if len(outlier_grp) else 0.0,
        "inlier_mean_delay_std": round(float(inlier_grp["delay_std"].mean()), 2) if len(inlier_grp) else 0.0,
    }

    alignment_metrics = {
        "kmeans_ari_vs_true_archetype": float(adjusted_rand_score(beh_train["true_archetype"], km_labels)),
        "kmeans_nmi_vs_true_archetype": float(normalized_mutual_info_score(beh_train["true_archetype"], km_labels)),
        "kmeans_silhouette": float(silhouette_score(X_scaled, km_labels)),
    }

    # 3. Quantify how clustering improves customer segmentation & downstream risk prediction
    segmentation_lift = _evaluate_segmentation_lift(
        datasets=datasets,
        beh_train=beh_train,
        scaler=scaler,
        kmeans=kmeans,
        dbscan=best_dbscan,
        X_train_scaled=X_scaled,
        best_eps=best_eps,
        tabular_cfg=tabular_cfg,
        seed=seed,
    )

    return CustomerClusteringArtifacts(
        scaler=scaler,
        kmeans=kmeans,
        dbscan=best_dbscan,
        dbscan_eps=best_eps,
        persona_map=persona_map,
        train_customer_assignments=beh_train,
        kmeans_k_sweep=k_sweep,
        dbscan_eps_sweep=eps_sweep,
        cluster_profiles=profiles,
        dbscan_summary=dbscan_summary,
        alignment_metrics=alignment_metrics,
        segmentation_lift=segmentation_lift,
    )


def assign_customer_cluster_features(
    meta: pd.DataFrame,
    X_tab: np.ndarray,
    tab_feature_names: list[str],
    beh_train: pd.DataFrame,
    scaler: StandardScaler,
    kmeans: KMeans,
    core_points_scaled: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Attach leakage-free behavioral cluster features to invoice rows using customer lookup or proxy."""
    # Lookup customer-level cluster features strictly from training customer behavior fit;
    # for unseen/fallback customers, derive approximate descriptor from point-in-time history.
    name_to_idx = {n: i for i, n in enumerate(tab_feature_names)}
    beh_lookup = beh_train.set_index("customer_id")
    k = kmeans.n_clusters

    extra_feats = np.zeros((len(meta), k + 2), dtype=np.float32)
    assigned_clusters = np.zeros(len(meta), dtype=np.int64)

    for row_i, cid in enumerate(meta["customer_id"].tolist()):
        if cid in beh_lookup.index:
            vec = beh_lookup.loc[cid, list(BEHAVIOR_CLUSTER_FEATURES)].to_numpy(dtype=np.float64)
            z = scaler.transform(vec.reshape(1, -1))
        else:
            # Point-in-time proxy if customer had no training records
            row = X_tab[row_i]
            proxy = np.asarray(
                [
                    row[name_to_idx["hist_delay_mean"]],
                    row[name_to_idx["hist_delay_std"]],
                    row[name_to_idx["hist_delay_slope"]],
                    row[name_to_idx["hist_early_rate"]],
                    row[name_to_idx["hist_late_rate"]],
                    row[name_to_idx["hist_delay_max"]],
                    max(row[name_to_idx["hist_tx_count_30d"]], 0.5),
                    row[name_to_idx["hist_tx_interval_std"]] / max(row[name_to_idx["hist_tx_interval_mean"]], 1.0),
                    row[name_to_idx["hist_log_amount_mean"]],
                    row[name_to_idx["hist_log_amount_std"]],
                    row[name_to_idx["hist_partial_rate"]],
                    row[name_to_idx["hist_avg_installments"]],
                    row[name_to_idx["hist_balance_util_mean"]],
                    row[name_to_idx["hist_balance_util_slope"]],
                ],
                dtype=np.float64,
            )
            z = scaler.transform(proxy.reshape(1, -1))

        dists = np.linalg.norm(kmeans.cluster_centers_ - z, axis=1)
        soft_assign = np.exp(-dists) / np.sum(np.exp(-dists))
        min_dist = float(np.min(dists))
        if len(core_points_scaled) > 0:
            core_dist = float(np.min(np.linalg.norm(core_points_scaled - z, axis=1)))
            is_outlier = 1.0 if core_dist > eps else 0.0
        else:
            is_outlier = 0.0

        assigned_clusters[row_i] = int(np.argmin(dists))
        extra_feats[row_i, :k] = soft_assign.astype(np.float32)
        extra_feats[row_i, k] = np.float32(min_dist)
        extra_feats[row_i, k + 1] = np.float32(is_outlier)

    return extra_feats, assigned_clusters


def _evaluate_segmentation_lift(
    datasets: PreparedDatasets,
    beh_train: pd.DataFrame,
    scaler: StandardScaler,
    kmeans: KMeans,
    dbscan: DBSCAN,
    X_train_scaled: np.ndarray,
    best_eps: float,
    tabular_cfg: TabularModelsConfig,
    seed: int,
) -> dict[str, Any]:
    core_idx = getattr(dbscan, "core_sample_indices_", np.array([], dtype=int))
    core_points_scaled = X_train_scaled[core_idx] if len(core_idx) > 0 else np.empty((0, X_train_scaled.shape[1]))

    tr_extra, _ = assign_customer_cluster_features(
        datasets.train.meta, datasets.train.X_tab, datasets.tab_feature_names,
        beh_train, scaler, kmeans, core_points_scaled, best_eps,
    )
    va_extra, _ = assign_customer_cluster_features(
        datasets.val.meta, datasets.val.X_tab, datasets.tab_feature_names,
        beh_train, scaler, kmeans, core_points_scaled, best_eps,
    )
    te_extra, te_clusters = assign_customer_cluster_features(
        datasets.test.meta, datasets.test.X_tab, datasets.tab_feature_names,
        beh_train, scaler, kmeans, core_points_scaled, best_eps,
    )

    # Compare baseline vs cluster-augmented Logistic Regression and XGBoost
    comparison: dict[str, Any] = {}
    for model_name, cls in [("logistic_regression", LogisticRegressionSuite), ("xgboost", XGBoostSuite)]:
        base_m = cls(tabular_cfg, seed=seed).fit(
            datasets.train.X_tab, datasets.train.y_late, datasets.train.y_delay,
            datasets.val.X_tab, datasets.val.y_late, datasets.val.y_delay,
        )
        base_val = base_m.predict(datasets.val.X_tab)
        thr_base = find_optimal_f1_threshold(datasets.val.y_late, base_val.probs)
        base_test = base_m.predict(datasets.test.X_tab)
        base_res = evaluate_predictions(
            datasets.test.y_late, datasets.test.y_delay, base_test.probs, base_test.delay_preds,
            threshold=thr_base, n_bootstrap=0,
        )

        aug_m = cls(tabular_cfg, seed=seed).fit(
            np.hstack([datasets.train.X_tab, tr_extra]), datasets.train.y_late, datasets.train.y_delay,
            np.hstack([datasets.val.X_tab, va_extra]), datasets.val.y_late, datasets.val.y_delay,
        )
        aug_val = aug_m.predict(np.hstack([datasets.val.X_tab, va_extra]))
        thr_aug = find_optimal_f1_threshold(datasets.val.y_late, aug_val.probs)
        aug_test = aug_m.predict(np.hstack([datasets.test.X_tab, te_extra]))
        aug_res = evaluate_predictions(
            datasets.test.y_late, datasets.test.y_delay, aug_test.probs, aug_test.delay_preds,
            threshold=thr_aug, n_bootstrap=0,
        )

        comparison[model_name] = {
            "baseline_roc_auc": base_res["roc_auc"],
            "cluster_augmented_roc_auc": aug_res["roc_auc"],
            "delta_roc_auc": aug_res["roc_auc"] - base_res["roc_auc"],
            "baseline_pr_auc": base_res["pr_auc"],
            "cluster_augmented_pr_auc": aug_res["pr_auc"],
            "delta_pr_auc": aug_res["pr_auc"] - base_res["pr_auc"],
            "baseline_ece": base_res["calibration_ece"],
            "cluster_augmented_ece": aug_res["calibration_ece"],
        }

    # Test-set risk stratification by discovered segment
    test_seg_rates = {}
    for cid in sorted(np.unique(te_clusters)):
        m = te_clusters == cid
        test_seg_rates[f"cluster_{int(cid)}"] = {
            "n_test_invoices": int(np.sum(m)),
            "test_late_rate": round(float(np.mean(datasets.test.y_late[m])), 4),
            "test_mean_delay_days": round(float(np.mean(datasets.test.y_delay[m])), 2),
        }

    return {
        "model_feature_lift": comparison,
        "test_segment_risk_stratification": test_seg_rates,
    }
