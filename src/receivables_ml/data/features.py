"""Leakage-free point-in-time temporal sequence and tabular feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from receivables_ml.config import DataSimulationConfig
from receivables_ml.data.generator import INDUSTRIES, REGIONS

SEQ_FEATURE_NAMES: tuple[str, ...] = (
    "days_since_prev_norm",
    "tx_rate_30d_norm",
    "observed_delay_norm",
    "delay_delta_norm",
    "is_late_observed",
    "is_settled_flag",
    "log_amount_norm",
    "amount_vs_hist_ratio",
    "partial_flag_observed",
    "installments_norm",
    "paid_ratio_at_t",
    "running_balance_log_norm",
    "running_utilization",
    "balance_delta_norm",
)

CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    "log_invoice_amount",
    "payment_terms_days",
    "credit_limit_log",
    "outstanding_balance_before_log",
    "balance_utilization",
    "balance_plus_new_utilization",
    "amount_to_limit_ratio",
    "open_invoices_before",
    "overdue_open_balance_ratio",
    "days_since_last_invoice",
    "issue_month_sin",
    "issue_month_cos",
) + tuple(f"industry_{ind}" for ind in INDUSTRIES) + tuple(f"region_{reg}" for reg in REGIONS)


@dataclass
class SequenceDatasetArrays:
    """Arrays for a single split (train / val / test)."""

    X_seq: np.ndarray
    seq_mask: np.ndarray
    seq_lengths: np.ndarray
    X_ctx: np.ndarray
    X_tab: np.ndarray
    y_late: np.ndarray
    y_delay: np.ndarray
    meta: pd.DataFrame


@dataclass
class PreparedDatasets:
    """Complete temporally split datasets and feature schemas."""

    train: SequenceDatasetArrays
    val: SequenceDatasetArrays
    test: SequenceDatasetArrays
    seq_feature_names: list[str]
    ctx_feature_names: list[str]
    tab_feature_names: list[str]
    customer_behavior_train: pd.DataFrame
    customer_behavior_all: pd.DataFrame
    split_cutoffs: dict[str, int]
    raw_transactions: pd.DataFrame


def _safe_slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y = np.asarray(values, dtype=np.float64)
    y_mean = y.mean()
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _encode_context(row: Any, days_since_last: float) -> np.ndarray:
    issue_ts = pd.Timestamp(row.issue_date)
    month_angle = 2.0 * np.pi * (issue_ts.month - 1) / 12.0
    out_bal = float(row.outstanding_balance_before)
    amt = float(row.invoice_amount)
    limit = float(row.credit_limit)
    overdue_ratio = float(row.overdue_open_balance_before) / max(out_bal, 1.0)

    vals = [
        np.log1p(amt),
        float(row.payment_terms_days),
        np.log1p(limit),
        np.log1p(out_bal),
        float(row.balance_utilization),
        (out_bal + amt) / max(limit, 1.0),
        amt / max(limit, 1.0),
        float(row.open_invoices_before),
        overdue_ratio,
        float(days_since_last),
        float(np.sin(month_angle)),
        float(np.cos(month_angle)),
    ]
    for ind in INDUSTRIES:
        vals.append(1.0 if row.industry == ind else 0.0)
    for reg in REGIONS:
        vals.append(1.0 if row.region == reg else 0.0)
    return np.asarray(vals, dtype=np.float32)


def compute_point_in_time_records(
    transactions: pd.DataFrame,
    config: DataSimulationConfig,
    dropout_rate: float = 0.0,
    dropout_seed: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    list[str],
]:
    """Build strictly causal point-in-time sequence, context, and tabular features.

    When scoring invoice i issued at day t_i, ONLY prior invoices k with issue_day_k < t_i
    are visible, and payment outcomes for k are capped to what is observable at t_i.
    Optional ``dropout_rate`` simulates missing historical transaction records.
    """
    rng = np.random.default_rng(dropout_seed) if dropout_rate > 0.0 else None
    max_len = config.max_seq_len
    n_seq_dim = len(SEQ_FEATURE_NAMES)

    df = transactions.sort_values(["customer_id", "issue_day", "seq_idx"]).reset_index(drop=True)

    tab_history_names = [
        "hist_count",
        "hist_settled_count",
        "hist_open_count",
        "hist_delay_mean",
        "hist_delay_std",
        "hist_delay_max",
        "hist_delay_last",
        "hist_delay_ewm",
        "hist_delay_slope",
        "hist_late_rate",
        "hist_early_rate",
        "hist_tx_interval_mean",
        "hist_tx_interval_std",
        "hist_tx_count_30d",
        "hist_tx_count_90d",
        "hist_log_amount_mean",
        "hist_log_amount_std",
        "amt_vs_hist_mean_ratio",
        "hist_partial_rate",
        "hist_avg_installments",
        "hist_unpaid_ratio_mean",
        "hist_balance_util_mean",
        "hist_balance_util_last",
        "hist_balance_util_slope",
    ]
    tab_feature_names = list(CONTEXT_FEATURE_NAMES) + tab_history_names

    seq_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    len_list: list[int] = []
    ctx_list: list[np.ndarray] = []
    tab_list: list[np.ndarray] = []
    y_late_list: list[int] = []
    y_delay_list: list[float] = []
    meta_rows: list[dict[str, Any]] = []

    for _, cust_df in df.groupby("customer_id", sort=False):
        rows = list(cust_df.itertuples(index=False))
        for idx, row in enumerate(rows):
            if idx < config.min_history_len:
                continue

            t_now = float(row.issue_day)
            prior_rows = rows[:idx]

            # Simulate missing historical transaction records (at least keep 1 if possible)
            if rng is not None and dropout_rate > 0.0 and len(prior_rows) > 1:
                keep_flags = rng.random(len(prior_rows)) >= dropout_rate
                if not np.any(keep_flags):
                    keep_flags[-1] = True
                prior_rows = [r for r, keep in zip(prior_rows, keep_flags) if keep]

            days_since_last = t_now - float(prior_rows[-1].issue_day)
            ctx_vec = _encode_context(row, days_since_last=days_since_last)

            # Construct observable causal sequence over prior_rows[-max_len:]
            window = prior_rows[-max_len:]
            seq_mat = np.zeros((max_len, n_seq_dim), dtype=np.float32)
            mask_vec = np.zeros(max_len, dtype=np.float32)

            obs_delays: list[float] = []
            obs_deltas: list[float] = []
            obs_log_amts: list[float] = []
            obs_partials: list[float] = []
            obs_insts: list[float] = []
            obs_unpaid_ratios: list[float] = []
            obs_utils: list[float] = []
            obs_settled: list[float] = []
            tx_30d_all = sum(1 for r in prior_rows if t_now - float(r.issue_day) <= 30.0)
            tx_90d_all = sum(1 for r in prior_rows if t_now - float(r.issue_day) <= 90.0)

            # Compute summaries across ALL visible prior rows for tabular aggregates,
            # and build tensor on the most recent `max_len` steps.
            running_amt_sum = 0.0
            prev_obs_delay = 0.0
            prev_util = 0.0
            seq_steps: list[np.ndarray] = []

            for k, pr in enumerate(prior_rows):
                settled = float(pr.final_paid_day) <= t_now
                if settled:
                    obs_delay = float(pr.payment_delay_days)
                    paid_ratio = 1.0
                    partial_obs = float(pr.is_partial_payment)
                    inst_obs = float(pr.n_installments)
                else:
                    # Strictly observable as of t_now without peeking into future final_paid_day
                    obs_delay = float(np.clip(t_now - float(pr.due_day), -15.0, 90.0))
                    if float(pr.is_partial_payment) == 1.0 and float(pr.first_paid_day) <= t_now:
                        paid_ratio = float(pr.first_paid_ratio)
                        partial_obs = 1.0
                        inst_obs = 2.0
                    else:
                        paid_ratio = 0.0
                        partial_obs = 0.0
                        inst_obs = 1.0

                dt = float(pr.issue_day - prior_rows[k - 1].issue_day) if k > 0 else float(pr.delta_days)
                log_amt = float(np.log1p(pr.invoice_amount))
                running_mean_amt = (running_amt_sum / k) if k > 0 else float(pr.invoice_amount)
                amt_ratio = float(pr.invoice_amount) / max(running_mean_amt, 1.0)
                running_amt_sum += float(pr.invoice_amount)

                util = float(pr.balance_utilization)
                delay_delta = obs_delay - prev_obs_delay if k > 0 else 0.0
                util_delta = util - prev_util if k > 0 else 0.0
                prev_obs_delay = obs_delay
                prev_util = util

                # 30-day rolling count up to pr.issue_day
                pr_30d = sum(
                    1
                    for prev_r in prior_rows[: k + 1]
                    if 0.0 <= float(pr.issue_day - prev_r.issue_day) <= 30.0
                )

                obs_delays.append(obs_delay)
                obs_deltas.append(dt)
                obs_log_amts.append(log_amt)
                obs_partials.append(partial_obs)
                obs_insts.append(inst_obs)
                obs_unpaid_ratios.append(1.0 - paid_ratio)
                obs_utils.append(util)
                obs_settled.append(1.0 if settled else 0.0)

                step_vec = np.asarray(
                    [
                        np.clip(dt / 30.0, 0.0, 4.0),
                        np.clip(pr_30d / 4.0, 0.0, 3.0),
                        np.clip(obs_delay / 30.0, -0.6, 3.0),
                        np.clip(delay_delta / 30.0, -3.0, 3.0),
                        1.0 if obs_delay > config.late_threshold_days else 0.0,
                        1.0 if settled else 0.0,
                        (log_amt - 9.0) / 1.5,
                        np.clip(amt_ratio - 1.0, -1.0, 3.0),
                        partial_obs,
                        (inst_obs - 1.0) / 3.0,
                        paid_ratio,
                        (np.log1p(float(pr.outstanding_balance_before)) - 9.0) / 2.5,
                        np.clip(util, 0.0, 3.0),
                        np.clip(util_delta, -2.0, 2.0),
                    ],
                    dtype=np.float32,
                )
                seq_steps.append(step_vec)

            recent_steps = seq_steps[-max_len:]
            L = len(recent_steps)
            seq_mat[:L] = np.stack(recent_steps, axis=0)
            mask_vec[:L] = 1.0

            # Exponentially weighted moving average of observed delays (recency emphasis)
            ewm_weights = np.exp(np.linspace(-1.8, 0.0, len(obs_delays)))
            ewm_weights /= ewm_weights.sum()
            delay_ewm = float(np.dot(obs_delays, ewm_weights))

            hist_mean_amt = float(np.exp(np.mean(obs_log_amts)) - 1.0)
            amt_vs_hist_mean_ratio = float(row.invoice_amount) / max(hist_mean_amt, 1.0)

            settled_count = float(np.sum(obs_settled))
            hist_feats = np.asarray(
                [
                    float(len(prior_rows)),
                    settled_count,
                    float(len(prior_rows) - settled_count),
                    float(np.mean(obs_delays)),
                    float(np.std(obs_delays)),
                    float(np.max(obs_delays)),
                    float(obs_delays[-1]),
                    delay_ewm,
                    _safe_slope(obs_delays[-8:]),
                    float(np.mean([1.0 if d > config.late_threshold_days else 0.0 for d in obs_delays])),
                    float(np.mean([1.0 if d <= 0.0 else 0.0 for d in obs_delays])),
                    float(np.mean(obs_deltas)),
                    float(np.std(obs_deltas)),
                    float(tx_30d_all),
                    float(tx_90d_all),
                    float(np.mean(obs_log_amts)),
                    float(np.std(obs_log_amts)),
                    amt_vs_hist_mean_ratio,
                    float(np.mean(obs_partials)),
                    float(np.mean(obs_insts)),
                    float(np.mean(obs_unpaid_ratios)),
                    float(np.mean(obs_utils)),
                    float(obs_utils[-1]),
                    _safe_slope(obs_utils[-8:]),
                ],
                dtype=np.float32,
            )

            seq_list.append(seq_mat)
            mask_list.append(mask_vec)
            len_list.append(L)
            ctx_list.append(ctx_vec)
            tab_list.append(np.concatenate([ctx_vec, hist_feats], axis=0))
            y_late_list.append(int(row.is_late))
            y_delay_list.append(float(row.payment_delay_days))
            meta_rows.append(
                {
                    "invoice_id": row.invoice_id,
                    "customer_id": row.customer_id,
                    "seq_idx": int(row.seq_idx),
                    "issue_day": int(row.issue_day),
                    "issue_date": row.issue_date,
                    "industry": row.industry,
                    "region": row.region,
                    "true_archetype": row.true_archetype,
                    "invoice_amount": float(row.invoice_amount),
                    "credit_limit": float(row.credit_limit),
                    "outstanding_balance_before": float(row.outstanding_balance_before),
                    "balance_utilization": float(row.balance_utilization),
                    "hist_delay_mean": float(np.mean(obs_delays)),
                    "hist_delay_last": float(obs_delays[-1]),
                    "hist_delay_slope": _safe_slope(obs_delays[-8:]),
                    "hist_partial_rate": float(np.mean(obs_partials)),
                    "hist_tx_interval_mean": float(np.mean(obs_deltas)),
                    "recent_delays": [round(float(d), 1) for d in obs_delays[-6:]],
                    "recent_balances": [round(float(pr.outstanding_balance_before), 2) for pr in window[-6:]],
                }
            )

    return (
        np.stack(seq_list, axis=0),
        np.stack(mask_list, axis=0),
        np.asarray(len_list, dtype=np.int64),
        np.stack(ctx_list, axis=0),
        np.stack(tab_list, axis=0),
        np.asarray(y_late_list, dtype=np.int64),
        np.asarray(y_delay_list, dtype=np.float32),
        pd.DataFrame(meta_rows),
        tab_feature_names,
    )


def build_customer_behavior_table(
    transactions: pd.DataFrame,
    late_threshold_days: int = 7,
    max_issue_day: int | None = None,
) -> pd.DataFrame:
    """Aggregate customer-level temporal behavioral descriptors up to optional cutoff."""
    df = transactions.copy()
    if max_issue_day is not None:
        df = df[df["issue_day"] <= max_issue_day].copy()
    df = df.sort_values(["customer_id", "issue_day", "seq_idx"])

    rows: list[dict[str, Any]] = []
    for cid, grp in df.groupby("customer_id", sort=True):
        if len(grp) < 2:
            continue
        delays = grp["payment_delay_days"].to_numpy(dtype=np.float64)
        deltas = grp["delta_days"].to_numpy(dtype=np.float64)
        log_amts = np.log1p(grp["invoice_amount"].to_numpy(dtype=np.float64))
        utils = grp["balance_utilization"].to_numpy(dtype=np.float64)
        partials = grp["is_partial_payment"].to_numpy(dtype=np.float64)
        insts = grp["n_installments"].to_numpy(dtype=np.float64)

        span_months = max((float(grp["issue_day"].max() - grp["issue_day"].min()) + 15.0) / 30.0, 1.0)
        tx_freq = len(grp) / span_months
        inter_cv = float(np.std(deltas) / max(np.mean(deltas), 1.0))

        rows.append(
            {
                "customer_id": cid,
                "true_archetype": str(grp["true_archetype"].iloc[0]),
                "industry": str(grp["industry"].iloc[0]),
                "n_tx": int(len(grp)),
                "delay_mean": float(np.mean(delays)),
                "delay_std": float(np.std(delays)),
                "delay_slope": _safe_slope(delays.tolist()),
                "early_pay_rate": float(np.mean(delays <= 0.0)),
                "late_rate": float(np.mean(delays > late_threshold_days)),
                "max_delay": float(np.max(delays)),
                "tx_frequency_per_month": float(tx_freq),
                "inter_invoice_cv": inter_cv,
                "log_amount_mean": float(np.mean(log_amts)),
                "log_amount_std": float(np.std(log_amts)),
                "partial_payment_rate": float(np.mean(partials)),
                "avg_installments": float(np.mean(insts)),
                "avg_balance_utilization": float(np.mean(utils)),
                "balance_utilization_slope": _safe_slope(utils.tolist()),
            }
        )
    return pd.DataFrame(rows)


def _slice_arrays(
    X_seq: np.ndarray,
    seq_mask: np.ndarray,
    seq_lengths: np.ndarray,
    X_ctx: np.ndarray,
    X_tab: np.ndarray,
    y_late: np.ndarray,
    y_delay: np.ndarray,
    meta: pd.DataFrame,
    mask: np.ndarray,
) -> SequenceDatasetArrays:
    return SequenceDatasetArrays(
        X_seq=X_seq[mask],
        seq_mask=seq_mask[mask],
        seq_lengths=seq_lengths[mask],
        X_ctx=X_ctx[mask],
        X_tab=X_tab[mask],
        y_late=y_late[mask],
        y_delay=y_delay[mask],
        meta=meta.loc[mask].reset_index(drop=True),
    )


def prepare_datasets(
    transactions: pd.DataFrame,
    config: DataSimulationConfig,
) -> PreparedDatasets:
    """Build point-in-time datasets and split chronologically into train/val/test."""
    (
        X_seq,
        seq_mask,
        seq_lengths,
        X_ctx,
        X_tab,
        y_late,
        y_delay,
        meta,
        tab_feature_names,
    ) = compute_point_in_time_records(transactions=transactions, config=config)

    days = meta["issue_day"].to_numpy()
    train_cutoff = int(np.quantile(days, config.train_quantile))
    val_cutoff = int(np.quantile(days, config.val_quantile))

    train_mask = days <= train_cutoff
    val_mask = (days > train_cutoff) & (days <= val_cutoff)
    test_mask = days > val_cutoff

    train_ds = _slice_arrays(X_seq, seq_mask, seq_lengths, X_ctx, X_tab, y_late, y_delay, meta, train_mask)
    val_ds = _slice_arrays(X_seq, seq_mask, seq_lengths, X_ctx, X_tab, y_late, y_delay, meta, val_mask)
    test_ds = _slice_arrays(X_seq, seq_mask, seq_lengths, X_ctx, X_tab, y_late, y_delay, meta, test_mask)

    behavior_train = build_customer_behavior_table(
        transactions=transactions,
        late_threshold_days=config.late_threshold_days,
        max_issue_day=train_cutoff,
    )
    behavior_all = build_customer_behavior_table(
        transactions=transactions,
        late_threshold_days=config.late_threshold_days,
        max_issue_day=None,
    )

    return PreparedDatasets(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        seq_feature_names=list(SEQ_FEATURE_NAMES),
        ctx_feature_names=list(CONTEXT_FEATURE_NAMES),
        tab_feature_names=tab_feature_names,
        customer_behavior_train=behavior_train,
        customer_behavior_all=behavior_all,
        split_cutoffs={"train_max_day": train_cutoff, "val_max_day": val_cutoff},
        raw_transactions=transactions,
    )
