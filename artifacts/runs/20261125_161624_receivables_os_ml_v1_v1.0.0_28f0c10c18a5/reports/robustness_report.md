# ReceivablesOS ML Robustness & Model Weakness Report

## 1. Executive Summary

All four trained models (**Logistic Regression**, **Random Forest**, **XGBoost**, and **Sequence Model (GRU+Attention)**) were stress-tested **without retraining** across five real-world failure modes: systemic payment delay shifts, industry mix shift, customer behavioral drift, missing historical records, and sudden correlated economic shocks.

## 2. Clean Holdout Test Baseline

| Model | ROC-AUC | PR-AUC | F1 | ECE (Calib) | Brier | MAE (days) | RMSE (days) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9185 | 0.9139 | 0.8802 | 0.0719 | 0.1109 | 6.39 | 9.23 |
| Random Forest | 0.9285 | 0.9259 | 0.8924 | 0.0546 | 0.1010 | 6.21 | 9.29 |
| XGBoost | 0.9324 | 0.9313 | 0.8895 | 0.0500 | 0.1035 | 6.05 | 8.98 |
| Sequence Model (GRU+Attn) | 0.9353 | 0.9327 | 0.8886 | 0.0289 | 0.0936 | 6.28 | 9.27 |

## 3. Model-by-Model Vulnerability Diagnosis

### Logistic Regression
- **Worst Discrimination Drop**: `-0.2969` ROC-AUC under *Increased Payment Delays (+20d)*
- **Worst Calibration Drift**: `+0.3251` ECE under *Increased Payment Delays (+20d)*
- **Worst Delay Regression Degradation**: `+12.21` MAE days under *Increased Payment Delays (+20d)*
- **Finding**: Severe calibration drift (+0.325 ECE) under 'Increased Payment Delays (+20d)', indicating uncalibrated probabilities under regime shift.
- **Finding**: Delay regression error spikes by +12.21 MAE days under 'Increased Payment Delays (+20d)', because static historical features/weights lag rapid systemic shocks.
- **Finding**: Sensitive to missing transaction histories (up to -0.022 ROC-AUC drop at 60% record dropout).

### Sequence Model (GRU+Attn)
- **Worst Discrimination Drop**: `-0.2931` ROC-AUC under *Increased Payment Delays (+20d)*
- **Worst Calibration Drift**: `+0.4073` ECE under *Increased Payment Delays (+20d)*
- **Worst Delay Regression Degradation**: `+14.06` MAE days under *Increased Payment Delays (+20d)*
- **Finding**: Severe calibration drift (+0.407 ECE) under 'Increased Payment Delays (+20d)', indicating uncalibrated probabilities under regime shift.
- **Finding**: Delay regression error spikes by +14.06 MAE days under 'Increased Payment Delays (+20d)', because static historical features/weights lag rapid systemic shocks.

### XGBoost
- **Worst Discrimination Drop**: `-0.2866` ROC-AUC under *Increased Payment Delays (+20d)*
- **Worst Calibration Drift**: `+0.4251` ECE under *Increased Payment Delays (+20d)*
- **Worst Delay Regression Degradation**: `+11.85` MAE days under *Increased Payment Delays (+20d)*
- **Finding**: Severe calibration drift (+0.425 ECE) under 'Increased Payment Delays (+20d)', indicating uncalibrated probabilities under regime shift.
- **Finding**: Delay regression error spikes by +11.85 MAE days under 'Increased Payment Delays (+20d)', because static historical features/weights lag rapid systemic shocks.

### Random Forest
- **Worst Discrimination Drop**: `-0.2887` ROC-AUC under *Increased Payment Delays (+20d)*
- **Worst Calibration Drift**: `+0.3553` ECE under *Increased Payment Delays (+20d)*
- **Worst Delay Regression Degradation**: `+12.65` MAE days under *Increased Payment Delays (+20d)*
- **Finding**: Severe calibration drift (+0.355 ECE) under 'Increased Payment Delays (+20d)', indicating uncalibrated probabilities under regime shift.
- **Finding**: Delay regression error spikes by +12.65 MAE days under 'Increased Payment Delays (+20d)', because static historical features/weights lag rapid systemic shocks.

## 4. Detailed Stress Test Matrix

| Scenario | Model | ROC-AUC (Delta) | PR-AUC (Delta) | ECE (Delta) | MAE (Delta) | RMSE (Delta) |
|---|---|---:|---:|---:|---:|---:|
| Increased Payment Delays (+5d) | Logistic Regression | 0.9424 (+0.0239) | 0.9585 (+0.0446) | 0.0513 (-0.0206) | 7.19 (+0.80) | 10.38 (+1.15) |
| Increased Payment Delays (+5d) | Random Forest | 0.9525 (+0.0240) | 0.9693 (+0.0434) | 0.1259 (+0.0713) | 8.28 (+2.07) | 11.33 (+2.03) |
| Increased Payment Delays (+5d) | XGBoost | 0.9556 (+0.0232) | 0.9707 (+0.0393) | 0.1251 (+0.0751) | 7.89 (+1.84) | 10.84 (+1.85) |
| Increased Payment Delays (+5d) | Sequence Model (GRU+Attn) | 0.9577 (+0.0224) | 0.9724 (+0.0397) | 0.0918 (+0.0630) | 8.25 (+1.98) | 11.30 (+2.03) |
| Increased Payment Delays (+10d) | Logistic Regression | 0.9084 (-0.0102) | 0.9680 (+0.0541) | 0.1857 (+0.1138) | 10.25 (+3.86) | 13.10 (+3.87) |
| Increased Payment Delays (+10d) | Random Forest | 0.9053 (-0.0232) | 0.9699 (+0.0440) | 0.2364 (+0.1817) | 10.97 (+4.77) | 14.17 (+4.88) |
| Increased Payment Delays (+10d) | XGBoost | 0.9075 (-0.0249) | 0.9705 (+0.0391) | 0.2621 (+0.2121) | 10.52 (+4.47) | 13.60 (+4.62) |
| Increased Payment Delays (+10d) | Sequence Model (GRU+Attn) | 0.9230 (-0.0123) | 0.9749 (+0.0422) | 0.2261 (+0.1972) | 11.69 (+5.41) | 14.47 (+5.20) |
| Increased Payment Delays (+20d) | Logistic Regression | 0.6216 (-0.2969) | 0.9966 (+0.0827) | 0.3970 (+0.3251) | 18.61 (+12.21) | 20.77 (+11.54) |
| Increased Payment Delays (+20d) | Random Forest | 0.6398 (-0.2887) | 0.9971 (+0.0712) | 0.4099 (+0.3553) | 18.85 (+12.65) | 21.57 (+12.27) |
| Increased Payment Delays (+20d) | XGBoost | 0.6458 (-0.2866) | 0.9971 (+0.0658) | 0.4751 (+0.4251) | 17.90 (+11.85) | 20.74 (+11.76) |
| Increased Payment Delays (+20d) | Sequence Model (GRU+Attn) | 0.6422 (-0.2931) | 0.9970 (+0.0643) | 0.4362 (+0.4073) | 20.34 (+14.06) | 22.47 (+13.20) |
| Industry Shift (Construction/Logistics Surge) | Logistic Regression | 0.9111 (-0.0074) | 0.9331 (+0.0192) | 0.0715 (-0.0004) | 6.87 (+0.48) | 9.60 (+0.37) |
| Industry Shift (Construction/Logistics Surge) | Random Forest | 0.9290 (+0.0005) | 0.9449 (+0.0190) | 0.0644 (+0.0097) | 6.79 (+0.59) | 9.95 (+0.65) |
| Industry Shift (Construction/Logistics Surge) | XGBoost | 0.9333 (+0.0009) | 0.9539 (+0.0225) | 0.0542 (+0.0042) | 6.60 (+0.55) | 9.55 (+0.57) |
| Industry Shift (Construction/Logistics Surge) | Sequence Model (GRU+Attn) | 0.9370 (+0.0017) | 0.9502 (+0.0176) | 0.0214 (-0.0075) | 6.86 (+0.58) | 9.85 (+0.58) |
| Customer Behavior Drift (35% Stable Cohort) | Logistic Regression | 0.8860 (-0.0325) | 0.9080 (-0.0059) | 0.0883 (+0.0164) | 7.40 (+1.01) | 10.03 (+0.80) |
| Customer Behavior Drift (35% Stable Cohort) | Random Forest | 0.9018 (-0.0266) | 0.9217 (-0.0042) | 0.1032 (+0.0485) | 7.54 (+1.33) | 10.34 (+1.04) |
| Customer Behavior Drift (35% Stable Cohort) | XGBoost | 0.9045 (-0.0278) | 0.9262 (-0.0051) | 0.1020 (+0.0521) | 7.31 (+1.26) | 9.98 (+1.00) |
| Customer Behavior Drift (35% Stable Cohort) | Sequence Model (GRU+Attn) | 0.9065 (-0.0288) | 0.9287 (-0.0040) | 0.0706 (+0.0418) | 7.59 (+1.31) | 10.31 (+1.04) |
| Missing Transaction Records (20% Dropout) | Logistic Regression | 0.9155 (-0.0031) | 0.9031 (-0.0108) | 0.0752 (+0.0032) | 6.37 (-0.02) | 9.27 (+0.04) |
| Missing Transaction Records (20% Dropout) | Random Forest | 0.9282 (-0.0003) | 0.9265 (+0.0006) | 0.0864 (+0.0317) | 6.28 (+0.08) | 9.36 (+0.06) |
| Missing Transaction Records (20% Dropout) | XGBoost | 0.9281 (-0.0043) | 0.9266 (-0.0048) | 0.0518 (+0.0018) | 6.23 (+0.18) | 9.15 (+0.17) |
| Missing Transaction Records (20% Dropout) | Sequence Model (GRU+Attn) | 0.9341 (-0.0012) | 0.9325 (-0.0002) | 0.0216 (-0.0072) | 6.35 (+0.07) | 9.29 (+0.03) |
| Missing Transaction Records (40% Dropout) | Logistic Regression | 0.9079 (-0.0106) | 0.8898 (-0.0241) | 0.0810 (+0.0091) | 6.50 (+0.10) | 9.40 (+0.17) |
| Missing Transaction Records (40% Dropout) | Random Forest | 0.9240 (-0.0045) | 0.9206 (-0.0053) | 0.1149 (+0.0603) | 6.34 (+0.14) | 9.36 (+0.06) |
| Missing Transaction Records (40% Dropout) | XGBoost | 0.9213 (-0.0111) | 0.9137 (-0.0177) | 0.0412 (-0.0088) | 6.58 (+0.54) | 9.27 (+0.29) |
| Missing Transaction Records (40% Dropout) | Sequence Model (GRU+Attn) | 0.9279 (-0.0074) | 0.9260 (-0.0067) | 0.0397 (+0.0109) | 6.97 (+0.69) | 9.69 (+0.42) |
| Missing Transaction Records (60% Dropout) | Logistic Regression | 0.8961 (-0.0225) | 0.8738 (-0.0401) | 0.0962 (+0.0243) | 6.62 (+0.23) | 9.53 (+0.29) |
| Missing Transaction Records (60% Dropout) | Random Forest | 0.9206 (-0.0079) | 0.9126 (-0.0133) | 0.1243 (+0.0697) | 6.40 (+0.20) | 9.40 (+0.10) |
| Missing Transaction Records (60% Dropout) | XGBoost | 0.9197 (-0.0127) | 0.9142 (-0.0172) | 0.0340 (-0.0160) | 7.00 (+0.95) | 9.35 (+0.37) |
| Missing Transaction Records (60% Dropout) | Sequence Model (GRU+Attn) | 0.9192 (-0.0161) | 0.9176 (-0.0150) | 0.0811 (+0.0522) | 7.91 (+1.63) | 10.37 (+1.10) |
| Sudden Economic Shock (Delay + Balance + Partial-Pay Surge) | Logistic Regression | 0.8966 (-0.0220) | 0.9786 (+0.0647) | 0.2577 (+0.1858) | 13.71 (+7.31) | 16.66 (+7.43) |
| Sudden Economic Shock (Delay + Balance + Partial-Pay Surge) | Random Forest | 0.8681 (-0.0604) | 0.9749 (+0.0490) | 0.3246 (+0.2700) | 15.40 (+9.20) | 18.42 (+9.13) |
| Sudden Economic Shock (Delay + Balance + Partial-Pay Surge) | XGBoost | 0.8727 (-0.0596) | 0.9754 (+0.0441) | 0.3336 (+0.2836) | 14.98 (+8.94) | 17.99 (+9.01) |
| Sudden Economic Shock (Delay + Balance + Partial-Pay Surge) | Sequence Model (GRU+Attn) | 0.9040 (-0.0313) | 0.9810 (+0.0483) | 0.2955 (+0.2667) | 15.96 (+9.68) | 18.92 (+9.65) |

## 5. Recommended Engineering Mitigations

1. **Rolling Recalibration (Platt / Isotonic)**: Macro delay shifts and economic shocks inflate ECE faster than they degrade ROC-AUC; trigger automated weekly temperature scaling when observed open-invoice overdue rate drifts > 5 percentage points.
2. **Macro & Sector Exogenous Signals**: Tabular and sequence encoders trained on closed historical windows underestimate abrupt systemic jumps (`MAE` spikes under sudden shock); incorporate live macro/sector overdue rate features.
3. **Missing-History Resilience**: When historical transaction records drop out (`40%-60%`), explicitly pass missingness indicators and fallback credit-bureau/static priors rather than relying solely on lagged aggregates.
4. **Concept-Drift Triggers**: Monitor rolling customer delay slope (`hist_delay_slope`) and PSI/KS shift across behavioral clusters to trigger targeted retraining on recent 60-90 day windows.
