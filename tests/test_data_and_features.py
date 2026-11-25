"""Tests for longitudinal simulator, point-in-time leakage prevention, and temporal splits."""

from __future__ import annotations

import numpy as np

from receivables_ml.config import DataSimulationConfig
from receivables_ml.data import (
    compute_point_in_time_records,
    generate_receivables_dataset,
    prepare_datasets,
)


def test_deterministic_generation() -> None:
    cfg = DataSimulationConfig(n_customers=40, min_invoices_per_customer=10, max_invoices_per_customer=14)
    c1, t1 = generate_receivables_dataset(cfg, seed=7)
    c2, t2 = generate_receivables_dataset(cfg, seed=7)
    assert c1.equals(c2)
    assert t1.equals(t2)
    assert len(c1) == 40
    assert {"customer_id", "invoice_amount", "outstanding_balance_before", "payment_delay_days", "is_late"}.issubset(t1.columns)


def test_strict_point_in_time_no_future_leakage() -> None:
    cfg = DataSimulationConfig(n_customers=25, min_invoices_per_customer=10, max_invoices_per_customer=12)
    _, tx = generate_receivables_dataset(cfg, seed=11)

    xs1, sm1, sl1, xc1, xt1, _, _, meta1, _ = compute_point_in_time_records(tx, cfg)

    # Pick a cutoff day and corrupt ALL future payment outcomes occurring after cutoff
    cutoff = int(meta1["issue_day"].median())
    tx_corrupt = tx.copy()
    still_open_at_cutoff = tx_corrupt["final_paid_day"] > cutoff
    tx_corrupt.loc[still_open_at_cutoff, "payment_delay_days"] = 88.0
    tx_corrupt.loc[still_open_at_cutoff, "is_late"] = 1
    tx_corrupt.loc[still_open_at_cutoff, "final_paid_day"] = np.maximum(
        tx_corrupt.loc[still_open_at_cutoff, "final_paid_day"], cutoff + 50.0
    )
    # Also ensure partial first_paid_day after cutoff stays after cutoff
    later_partial = tx_corrupt["first_paid_day"] > cutoff
    tx_corrupt.loc[later_partial, "first_paid_day"] = tx_corrupt.loc[later_partial, "first_paid_day"] + 25.0

    xs2, sm2, sl2, xc2, xt2, _, _, meta2, _ = compute_point_in_time_records(tx_corrupt, cfg)

    early_mask = meta1["issue_day"].to_numpy() <= cutoff
    assert np.allclose(xs1[early_mask], xs2[early_mask], atol=1e-6)
    assert np.array_equal(sm1[early_mask], sm2[early_mask])
    assert np.array_equal(sl1[early_mask], sl2[early_mask])
    assert np.allclose(xc1[early_mask], xc2[early_mask], atol=1e-6)
    assert np.allclose(xt1[early_mask], xt2[early_mask], atol=1e-6)


def test_chronological_splits_and_dropout() -> None:
    cfg = DataSimulationConfig(n_customers=50, min_invoices_per_customer=10, max_invoices_per_customer=14)
    _, tx = generate_receivables_dataset(cfg, seed=42)
    ds = prepare_datasets(tx, cfg)

    assert ds.train.meta["issue_day"].max() <= ds.split_cutoffs["train_max_day"]
    assert ds.val.meta["issue_day"].min() > ds.split_cutoffs["train_max_day"]
    assert ds.val.meta["issue_day"].max() <= ds.split_cutoffs["val_max_day"]
    assert ds.test.meta["issue_day"].min() > ds.split_cutoffs["val_max_day"]

    # Dropout reduces observed sequence lengths cleanly without NaNs
    xs_full, _, sl_full, _, xt_full, _, _, _, _ = compute_point_in_time_records(tx, cfg, dropout_rate=0.0)
    xs_drop, sm_drop, sl_drop, _, xt_drop, _, _, _, _ = compute_point_in_time_records(tx, cfg, dropout_rate=0.5, dropout_seed=99)
    assert sl_drop.mean() < sl_full.mean()
    assert not np.isnan(xs_drop).any()
    assert not np.isnan(xt_drop).any()
    assert np.all((sm_drop.sum(axis=1) == sl_drop))
