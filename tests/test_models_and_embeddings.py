"""Tests for tabular models, PyTorch Temporal Sequence Model, embeddings, and 8 metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from receivables_ml.config import DataSimulationConfig, SequenceModelConfig, TabularModelsConfig
from receivables_ml.data import generate_receivables_dataset, prepare_datasets
from receivables_ml.evaluation.metrics import evaluate_predictions
from receivables_ml.models import SequenceRiskModel, build_tabular_models


def test_tabular_and_sequence_models_roundtrip(tmp_path: Path) -> None:
    d_cfg = DataSimulationConfig(n_customers=70, min_invoices_per_customer=10, max_invoices_per_customer=14)
    _, tx = generate_receivables_dataset(d_cfg, seed=42)
    ds = prepare_datasets(tx, d_cfg)

    tab_cfg = TabularModelsConfig(rf_n_estimators=30, xgb_n_estimators=40)
    models = build_tabular_models(tab_cfg, seed=42)
    for name, model in models.items():
        model.fit(ds.train.X_tab, ds.train.y_late, ds.train.y_delay, ds.val.X_tab, ds.val.y_late, ds.val.y_delay)
        pred = model.predict(ds.test.X_tab)
        assert pred.probs.shape == (len(ds.test.y_late),)
        assert np.all((pred.probs >= 0.0) & (pred.probs <= 1.0))
        assert pred.delay_preds.shape == (len(ds.test.y_delay),)
        metrics = evaluate_predictions(ds.test.y_late, ds.test.y_delay, pred.probs, pred.delay_preds, n_bootstrap=0)
        for key in ("roc_auc", "pr_auc", "precision", "recall", "f1", "calibration_ece", "brier_score", "mae", "rmse"):
            assert key in metrics
            assert np.isfinite(metrics[key])

        saved = model.save(tmp_path / name)
        assert saved.exists()

    seq_cfg = SequenceModelConfig(hidden_dim=24, embedding_dim=16, epochs=4, batch_size=64)
    seq_model = SequenceRiskModel(
        seq_input_dim=ds.train.X_seq.shape[-1],
        ctx_input_dim=ds.train.X_ctx.shape[-1],
        config=seq_cfg,
        seed=42,
    )
    seq_model.fit(
        ds.train.X_seq, ds.train.seq_mask, ds.train.seq_lengths, ds.train.X_ctx, ds.train.y_late, ds.train.y_delay,
        ds.val.X_seq, ds.val.seq_mask, ds.val.seq_lengths, ds.val.X_ctx, ds.val.y_late, ds.val.y_delay,
    )
    seq_pred = seq_model.predict(ds.test.X_seq, ds.test.seq_mask, ds.test.seq_lengths, ds.test.X_ctx)
    assert np.all((seq_pred.probs >= 0.0) & (seq_pred.probs <= 1.0))

    emb, attn = seq_model.extract_embeddings(ds.test.X_seq, ds.test.seq_mask, ds.test.seq_lengths)
    assert emb.shape == (len(ds.test.y_late), 16)
    assert np.allclose(attn.sum(axis=1), 1.0, atol=1e-4)
    # Padded positions receive zero attention mass
    padded_mask = ds.test.seq_mask == 0.0
    assert np.all(attn[padded_mask] < 1e-6)

    ckpt_path = seq_model.save(tmp_path / "seq")
    reloaded = SequenceRiskModel.load(ckpt_path)
    reloaded_pred = reloaded.predict(ds.test.X_seq, ds.test.seq_mask, ds.test.seq_lengths, ds.test.X_ctx)
    assert np.allclose(seq_pred.probs, reloaded_pred.probs, atol=1e-6)
