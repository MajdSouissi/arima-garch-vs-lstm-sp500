# Master Comparison — Final Thesis Results

**Generated:** 2026-09-05 10:25
**Test Period:** 2023-01-03 → 2026-08-13 (906 obs)

---

## A. Return Forecasting

Retained statistical model for returns: **GJR-GARCH(1,1) + Constant Mean + Student-t**
(selected by BIC from `mean_variance_report.md`; `GARCH_Const` in existing results)

| Model | Type | RMSE | MAE | DirAcc (%) |
|-------|------|------|-----|------------|
| GARCH_Const (GJR) | Statistical | 0.933423 | 0.655058 | 56.18 |
| LSTM_Base | Deep Learning | 0.930757 | 0.661675 | 49.45 |

### DM Tests (Returns)

| Loss | DM Stat | p-value | Conclusion |
|------|---------|---------|------------|
| SE | 0.2144 | 0.8303 | No significant difference (p=0.8303) |
| AE | -2.0087 | 0.0446 | GARCH_Const significantly better (p=0.0446) |

---

## B. Volatility Forecasting

| Model | Type | RMSE | MAE | QLIKE |
|-------|------|------|-----|-------|
| GARCH(1,1) symmetric | Statistical | 3.254202 | 1.017389 | 1.653101 |
| GJR-GARCH(1,1) (existing) | Statistical | 3.066211 | 0.946062 | 1.593644 |
| LSTM_Vol | Deep Learning | 3.203975 | 1.004294 | 1.577264 |

### DM Tests — GARCH(1,1) symmetric vs LSTM_Vol

| Loss | DM Stat | p-value | N | Conclusion |
|------|---------|---------|---|------------|
| QLIKE | 3.6770 | 0.0002 | 906 | LSTM_Vol significantly better (p=0.0002) |
| SE | 0.5879 | 0.5566 | 906 | No significant difference (p=0.5566) |

### DM Tests — GJR-GARCH(1,1) vs LSTM_Vol (existing)

| Loss | DM Stat | p-value | Conclusion |
|------|---------|---------|------------|
| QLIKE | 0.7322 | 0.4641 | No significant difference (p=0.4641) |
| SE | -2.0313 | 0.0422 | GARCH_Const significantly better (p=0.0422) |

---

## C. Methodological Notes

- All GARCH parameters estimated on Training set only (2000–2018)
- LSTM scalers fitted on Training set only; early stopping on Validation set
- No hyperparameter tuning on the Test set
- Chronological (non-shuffled) data split
- No outlier removal or data winsorization
- LSTM reproducible random seed = 42
- Volatility target: r²_t (squared log return) for both GARCH and LSTM
- GARCH forecasts σ_t; evaluation uses σ²_t vs r²_t
- LSTM_Vol directly forecasts r²_t

## D. Files

| File | Content |
|------|---------|
| `results/garch11_forecasts.csv` | GARCH(1,1) symmetric forecasts (full series) |
| `results/garch11_metrics.csv` | OOS metrics for GARCH(1,1) and LSTM (verification) |
| `results/garch11_dm_tests.csv` | DM tests: GARCH(1,1) vs LSTM_Vol |
| `results/garch11_report.md` | Full GARCH(1,1) evaluation report |
| `results/master_comparison.md` | This file — master synthesis |
| `results/figures/fig_covid_stress_volatility.png` | COVID-19 stress figure |
| `scripts/run_garch11_oos.py` | Reproducible script |