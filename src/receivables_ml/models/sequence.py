"""Temporal Customer Embeddings and PyTorch GRU+Attention Sequence Model."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from receivables_ml.config import SequenceModelConfig
from receivables_ml.evaluation.metrics import ModelPrediction


class TemporalGRUAttentionNet(nn.Module):
    """Multi-task GRU + temporal attention sequence network producing customer embeddings."""

    def __init__(
        self,
        seq_input_dim: int,
        ctx_input_dim: int,
        hidden_dim: int = 48,
        embedding_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(ctx_input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.cls_head = nn.Linear(hidden_dim // 2, 1)
        self.reg_head = nn.Linear(hidden_dim // 2, 1)

    def encode_sequence(
        self,
        x_seq: torch.Tensor,
        seq_mask: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode padded sequence into normalized temporal customer embedding + attention."""
        projected = self.seq_proj(x_seq)
        gru_out, _ = self.gru(projected)

        # Masked additive temporal attention
        raw_scores = self.attn_score(gru_out).squeeze(-1)
        raw_scores = raw_scores.masked_fill(seq_mask <= 0.0, -1e9)
        attn_weights = torch.softmax(raw_scores, dim=-1)
        attn_context = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)

        # Gather final valid sequence step hidden state
        last_idx = torch.clamp(seq_lengths - 1, min=0)
        batch_indices = torch.arange(x_seq.size(0), device=x_seq.device)
        last_hidden = gru_out[batch_indices, last_idx]

        combined = torch.cat([attn_context, last_hidden], dim=-1)
        embedding = self.embedding_head(combined)
        return embedding, attn_weights

    def forward(
        self,
        x_seq: torch.Tensor,
        seq_mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        x_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding, attn_weights = self.encode_sequence(x_seq, seq_mask, seq_lengths)
        ctx_repr = self.ctx_proj(x_ctx)
        fused = self.fusion(torch.cat([embedding, ctx_repr], dim=-1))
        risk_logit = self.cls_head(fused).squeeze(-1)
        delay_pred = self.reg_head(fused).squeeze(-1)
        return risk_logit, delay_pred, embedding, attn_weights


class SequenceRiskModel:
    """Sklearn-like trainer & inference wrapper for TemporalGRUAttentionNet."""

    name = "sequence_model"

    def __init__(
        self,
        seq_input_dim: int,
        ctx_input_dim: int,
        config: SequenceModelConfig,
        seed: int = 42,
    ) -> None:
        self.seq_input_dim = seq_input_dim
        self.ctx_input_dim = ctx_input_dim
        self.config = config
        self.seed = seed
        self.device = torch.device("cpu")

        torch.manual_seed(seed)
        self.net = TemporalGRUAttentionNet(
            seq_input_dim=seq_input_dim,
            ctx_input_dim=ctx_input_dim,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ).to(self.device)
        self.ctx_scaler = StandardScaler()
        self.train_history: list[dict[str, float]] = []

    def fit(
        self,
        X_seq_train: np.ndarray,
        seq_mask_train: np.ndarray,
        seq_len_train: np.ndarray,
        X_ctx_train: np.ndarray,
        y_late_train: np.ndarray,
        y_delay_train: np.ndarray,
        X_seq_val: np.ndarray,
        seq_mask_val: np.ndarray,
        seq_len_val: np.ndarray,
        X_ctx_val: np.ndarray,
        y_late_val: np.ndarray,
        y_delay_val: np.ndarray,
    ) -> "SequenceRiskModel":
        torch.manual_seed(self.seed)
        X_ctx_tr_scaled = self.ctx_scaler.fit_transform(X_ctx_train).astype(np.float32)
        X_ctx_va_scaled = self.ctx_scaler.transform(X_ctx_val).astype(np.float32)

        train_ds = TensorDataset(
            torch.tensor(X_seq_train, dtype=torch.float32),
            torch.tensor(seq_mask_train, dtype=torch.float32),
            torch.tensor(seq_len_train, dtype=torch.long),
            torch.tensor(X_ctx_tr_scaled, dtype=torch.float32),
            torch.tensor(y_late_train, dtype=torch.float32),
            torch.tensor(y_delay_train, dtype=torch.float32),
        )
        gen = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, generator=gen)

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        bce_loss_fn = nn.BCEWithLogitsLoss()
        reg_loss_fn = nn.SmoothL1Loss(beta=3.0)

        best_val_loss = float("inf")
        best_state = copy.deepcopy(self.net.state_dict())
        patience_left = self.config.early_stopping_patience
        self.train_history = []

        for epoch in range(1, self.config.epochs + 1):
            self.net.train()
            tr_losses = []
            for b_seq, b_mask, b_len, b_ctx, b_late, b_delay in loader:
                optimizer.zero_grad()
                logits, delay_hat, _, _ = self.net(b_seq, b_mask, b_len, b_ctx)
                loss_cls = bce_loss_fn(logits, b_late)
                loss_reg = reg_loss_fn(delay_hat, b_delay)
                loss = loss_cls + self.config.delay_loss_weight * loss_reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=2.0)
                optimizer.step()
                tr_losses.append(float(loss.item()))

            val_loss = self._eval_loss(
                X_seq_val,
                seq_mask_val,
                seq_len_val,
                X_ctx_va_scaled,
                y_late_val,
                y_delay_val,
                bce_loss_fn,
                reg_loss_fn,
            )
            self.train_history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(np.mean(tr_losses)),
                    "val_loss": float(val_loss),
                }
            )
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.net.state_dict())
                patience_left = self.config.early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        self.net.load_state_dict(best_state)
        return self

    def _eval_loss(
        self,
        X_seq: np.ndarray,
        seq_mask: np.ndarray,
        seq_len: np.ndarray,
        X_ctx_scaled: np.ndarray,
        y_late: np.ndarray,
        y_delay: np.ndarray,
        bce_fn: nn.Module,
        reg_fn: nn.Module,
    ) -> float:
        self.net.eval()
        with torch.no_grad():
            logits, delay_hat, _, _ = self.net(
                torch.tensor(X_seq, dtype=torch.float32),
                torch.tensor(seq_mask, dtype=torch.float32),
                torch.tensor(seq_len, dtype=torch.long),
                torch.tensor(X_ctx_scaled, dtype=torch.float32),
            )
            loss_cls = bce_fn(logits, torch.tensor(y_late, dtype=torch.float32))
            loss_reg = reg_fn(delay_hat, torch.tensor(y_delay, dtype=torch.float32))
            return float((loss_cls + self.config.delay_loss_weight * loss_reg).item())

    def predict(
        self,
        X_seq: np.ndarray,
        seq_mask: np.ndarray,
        seq_lengths: np.ndarray,
        X_ctx: np.ndarray,
    ) -> ModelPrediction:
        self.net.eval()
        X_ctx_scaled = self.ctx_scaler.transform(X_ctx).astype(np.float32)
        with torch.no_grad():
            logits, delay_hat, _, _ = self.net(
                torch.tensor(X_seq, dtype=torch.float32),
                torch.tensor(seq_mask, dtype=torch.float32),
                torch.tensor(seq_lengths, dtype=torch.long),
                torch.tensor(X_ctx_scaled, dtype=torch.float32),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            delay_preds = np.clip(delay_hat.cpu().numpy(), -15.0, 90.0)
        return ModelPrediction(probs=probs, delay_preds=delay_preds)

    def extract_embeddings(
        self,
        X_seq: np.ndarray,
        seq_mask: np.ndarray,
        seq_lengths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract learned temporal customer embeddings and attention weights."""
        self.net.eval()
        with torch.no_grad():
            emb, attn = self.net.encode_sequence(
                torch.tensor(X_seq, dtype=torch.float32),
                torch.tensor(seq_mask, dtype=torch.float32),
                torch.tensor(seq_lengths, dtype=torch.long),
            )
        return emb.cpu().numpy(), attn.cpu().numpy()

    def save(self, directory: str | Path) -> Path:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "sequence_model.pt"
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "ctx_scaler": self.ctx_scaler,
                "seq_input_dim": self.seq_input_dim,
                "ctx_input_dim": self.ctx_input_dim,
                "config": self.config.model_dump(),
                "seed": self.seed,
                "train_history": self.train_history,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SequenceRiskModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = SequenceModelConfig.model_validate(ckpt["config"])
        model = cls(
            seq_input_dim=int(ckpt["seq_input_dim"]),
            ctx_input_dim=int(ckpt["ctx_input_dim"]),
            config=cfg,
            seed=int(ckpt["seed"]),
        )
        model.net.load_state_dict(ckpt["state_dict"])
        model.ctx_scaler = ckpt["ctx_scaler"]
        model.train_history = ckpt.get("train_history", [])
        return model


def analyze_temporal_embeddings(
    train_embeddings: np.ndarray,
    train_meta: pd.DataFrame,
    test_embeddings: np.ndarray,
    test_attn: np.ndarray,
    test_meta: pd.DataFrame,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze geometric structure, archetype separability, and attention recency."""
    # Latest-invoice embedding per customer in test set
    test_df = test_meta[["customer_id", "seq_idx", "true_archetype", "hist_delay_mean", "balance_utilization"]].copy()
    test_df["row_idx"] = np.arange(len(test_df))
    latest_rows = test_df.sort_values("seq_idx").groupby("customer_id", as_index=False).tail(1)
    idx_latest = latest_rows["row_idx"].to_numpy()
    cust_emb = test_embeddings[idx_latest]
    cust_arch = latest_rows["true_archetype"].to_numpy()

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(cust_emb)

    # Linear probe: how well do frozen embeddings trained only for late-payment predict behavioral archetypes?
    train_df = train_meta[["customer_id", "seq_idx", "true_archetype"]].copy()
    train_df["row_idx"] = np.arange(len(train_df))
    tr_latest = train_df.sort_values("seq_idx").groupby("customer_id", as_index=False).tail(1)
    probe = LogisticRegression(max_iter=1000, random_state=seed)
    probe.fit(train_embeddings[tr_latest["row_idx"].to_numpy()], tr_latest["true_archetype"].to_numpy())
    probe_acc = float(accuracy_score(cust_arch, probe.predict(cust_emb)))

    sil = float(silhouette_score(cust_emb, cust_arch)) if len(np.unique(cust_arch)) > 1 else 0.0

    # Mean attention mass on most recent 3 steps vs older steps
    valid_steps = np.sum(test_attn > 1e-6, axis=1)
    recent3_mass = []
    for row_i, length in enumerate(valid_steps):
        if length > 0:
            start = max(0, int(length) - 3)
            recent3_mass.append(float(np.sum(test_attn[row_i, start:int(length)])))

    projection_records = []
    for i, rec in enumerate(latest_rows.itertuples(index=False)):
        projection_records.append(
            {
                "customer_id": rec.customer_id,
                "true_archetype": rec.true_archetype,
                "pca_1": float(coords[i, 0]),
                "pca_2": float(coords[i, 1]),
                "hist_delay_mean": float(rec.hist_delay_mean),
                "balance_utilization": float(rec.balance_utilization),
            }
        )

    return {
        "embedding_dim": int(test_embeddings.shape[1]),
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "archetype_linear_probe_accuracy": probe_acc,
        "archetype_silhouette_on_embeddings": sil,
        "mean_recent3_attention_mass": float(np.mean(recent3_mass)) if recent3_mass else 0.0,
        "customer_projection": projection_records,
    }
