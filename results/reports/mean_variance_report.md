# S&P 500 — Conditional Mean and Variance Structure Report

**Master's Research Project:**  
*Comparaison des modèles statistiques et d'apprentissage automatique
pour la prévision financière (ARIMA-GARCH et LSTM)*

**Date generated:** 2026-08-14 17:26

## GARCH Model Comparison

| Model | AIC | BIC | Log-Lik |
|-------|-----|-----|---------|
| const+GJR-GARCH(1,1)+t | 17312.56 | 17353.27 | -8650.28 |
| MA(1)+GJR-GARCH(1,1)+t | 17312.56 | 17353.27 | -8650.28 |
| const+GARCH(1,1)+t | 17539.34 | 17573.26 | -8764.67 |
| MA(1)+GARCH(1,1)+t | 17539.34 | 17573.26 | -8764.67 |
| const+EGARCH(1,1)+t | 17564.68 | 17598.60 | -8777.34 |
| MA(1)+EGARCH(1,1)+t | 17564.68 | 17598.60 | -8777.34 |
| const+GJR-GARCH(1,1)+normal | 17595.08 | 17629.00 | -8792.54 |
| MA(1)+GJR-GARCH(1,1)+normal | 17595.08 | 17629.00 | -8792.54 |
| const+GARCH(1,1)+normal | 17846.10 | 17873.24 | -8919.05 |
| MA(1)+GARCH(1,1)+normal | 17846.10 | 17873.24 | -8919.05 |
| const+EGARCH(1,1)+normal | 17905.58 | 17932.72 | -8948.79 |
| MA(1)+EGARCH(1,1)+normal | 17905.58 | 17932.72 | -8948.79 |

## KPSS Correction (from previous report)

The KPSS test for Log_Return has p = 0.0475. At α = 0.05, this is a
**borderline rejection** of H0 (level-stationarity), NOT a failure to reject.
The ADF test provides strong evidence against a unit root (p ≈ 0).
The borderline KPSS rejection likely reflects slowly evolving unconditional
variance (volatility clustering) rather than mean non-stationarity.
The return series is treated as mean-stationary for ARMA modeling.

## Research Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Conditional mean necessary? | Modest benefit; constant-mean is defensible |
| 2 | ARMA structure? | MA(1) (selected) or ARMA(0,0) (baseline) |
| 3 | ARCH after mean model? | Yes — highly significant at all lags |
| 4 | GARCH(1,1) reasonable? | Yes — standard baseline with high persistence |
| 5 | Asymmetric volatility? | Yes — leverage effect confirmed |
| 6 | Student-t worth testing? | Yes — residual excess kurtosis after GARCH |
| 7 | Recommended specs? | Top 3 by BIC (see table above) |

## Figures

| Figure | Description |
|--------|-------------|
| task1_acf_pacf_returns.png | ACF/PACF of log returns |
| task3_selected_mean_diagnostics.png | Residual diagnostics for selected mean model |
| task7_diagnostics_model1.png | Full diagnostics for best GARCH model |
| task7_diagnostics_model2.png | Full diagnostics for second-best GARCH model |

---
*End of conditional mean/variance report.*