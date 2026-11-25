# ReceivablesOS ML Intelligence Research Summary (`20261125_161624_receivables_os_ml_v1_v1.0.0_28f0c10c18a5`)

- **Model Version**: `1.0.0` | **Config Hash**: `28f0c10c18a5` | **Seed**: `42`
- **Customers**: `600` | **Total Transactions**: `11831` | **Splits (Train/Val/Test)**: `6537/1500/1994`

## 1. Model Comparison (Holdout Test Set)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | ECE (Calib) | Brier | MAE (days) | RMSE (days) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9185 | 0.9139 | 0.7973 | 0.9823 | 0.8802 | 0.0719 | 0.1109 | 6.39 | 9.23 |
| Random Forest | 0.9285 | 0.9259 | 0.8075 | 0.9972 | 0.8924 | 0.0546 | 0.1010 | 6.21 | 9.29 |
| XGBoost | 0.9324 | 0.9313 | 0.8122 | 0.9832 | 0.8895 | 0.0500 | 0.1035 | 6.05 | 8.98 |
| Sequence Model (GRU+Attn) | 0.9353 | 0.9327 | 0.8144 | 0.9776 | 0.8886 | 0.0289 | 0.0936 | 6.28 | 9.27 |

## 2. Temporal Customer Embeddings & Sequence vs XGBoost

- **Sequence Embedding Dim**: `32` | **Archetype Linear Probe Accuracy**: `0.9283` | **Recent-3 Step Attention Mass**: `0.155`
- **Hybrid Ablation (XGBoost + Temporal Embeddings vs XGBoost Tabular)**: `Delta ROC-AUC = +0.00363`, `Delta PR-AUC = +0.00267`, `Delta MAE = +0.0554` days

## 3. Customer Payment Behavior Clustering (K-Means + DBSCAN)

- **K-Means Alignment vs Latent Archetypes**: `ARI = 0.7686`, `NMI = 0.8198`, `Silhouette = 0.4064`
- **DBSCAN Density Outliers**: `184` customers (`30.67%`) with mean late rate `0.637` vs inliers `0.396`

| Segment Name | Customers | Mean Delay (d) | Delay Slope | Late Rate | Partial Pay Rate | Avg Util |
|---|---:|---:|---:|---:|---:|---:|
| Irregular Payers | 114 (19.0%) | 10.0 | -0.129 | 0.568 | 0.460 | 0.334 |
| Consistent Early Payers | 222 (37.0%) | -3.8 | -0.058 | 0.018 | 0.043 | 0.077 |
| Deteriorating Customers | 22 (3.67%) | 15.4 | 1.802 | 0.644 | 0.604 | 0.632 |
| Consistent Late Payers | 150 (25.0%) | 12.0 | 0.017 | 0.779 | 0.352 | 0.315 |
| High-Value Delayed Payers | 92 (15.33%) | 15.8 | -0.039 | 0.893 | 0.697 | 0.376 |

## 4. Collection Action Optimization (Contextual Bandit Simulation)

> **Disclaimer**: NON-CAUSAL SIMULATION NOTICE: Payment probabilities and expected collection rewards are estimated on synthetic historical episodes using off-policy evaluation (IPS, SNIPS, Doubly Robust) and a synthetic response model. They demonstrate algorithmic policy optimization mechanics and DO NOT prove real-world causal effectiveness. Randomized field experiments (A/B testing) are required prior to autonomous production rollout.

| Policy | Mean Cost ($) | Est. Pay Prob | OPE IPS | OPE SNIPS | OPE Doubly Robust | Sim Oracle Reward |
|---|---:|---:|---:|---:|---:|---:|
| Logging Policy (Historical Heuristic) | 4.13 | 0.7467 | 270.62 | 270.62 | 270.62 | 279.51 |
| Always Email | 0.15 | 0.6559 | 248.31 | 234.10 | 237.44 | 250.11 |
| Always WhatsApp | 0.40 | 0.7899 | 305.58 | 290.08 | 282.69 | 288.70 |
| Always Phone Call | 6.50 | 0.8445 | 286.01 | 316.49 | 317.06 | 331.39 |
| Always Escalation | 28.00 | 0.7617 | 151.83 | 180.04 | 184.59 | 188.20 |
| Direct-Method Outcome Policy | 4.26 | 0.8521 | 302.99 | 306.48 | 326.06 | 334.47 |
| LinUCB Contextual Bandit | 4.19 | 0.8499 | 307.18 | 307.45 | 326.84 | 334.25 |
| Doubly-Robust Policy Optimizer | 3.98 | 0.8485 | 302.84 | 308.20 | 328.22 | 334.31 |

## 5. SHAP Explainability Highlights

- **High-Risk Customers Identified**: `310` (threshold >= `0.55`), explained `100`
- **Top 5 Global SHAP Drivers (XGBoost)**: `hist_delay_max` (1.0613), `hist_late_rate` (0.5101), `hist_balance_util_mean` (0.2767), `outstanding_balance_before_log` (0.2190), `hist_unpaid_ratio_mean` (0.2144)

## 6. Robustness Summary

See `reports/robustness_report.md` for full 5-scenario stress test matrix and model weakness diagnoses.

- **Logistic Regression**: worst AUC drop `-0.2969` (Increased Payment Delays (+20d)), worst ECE drift `+0.3251` (Increased Payment Delays (+20d)), worst MAE jump `+12.21d`
- **Sequence Model (GRU+Attn)**: worst AUC drop `-0.2931` (Increased Payment Delays (+20d)), worst ECE drift `+0.4073` (Increased Payment Delays (+20d)), worst MAE jump `+14.06d`
- **XGBoost**: worst AUC drop `-0.2866` (Increased Payment Delays (+20d)), worst ECE drift `+0.4251` (Increased Payment Delays (+20d)), worst MAE jump `+11.85d`
- **Random Forest**: worst AUC drop `-0.2887` (Increased Payment Delays (+20d)), worst ECE drift `+0.3553` (Increased Payment Delays (+20d)), worst MAE jump `+12.65d`