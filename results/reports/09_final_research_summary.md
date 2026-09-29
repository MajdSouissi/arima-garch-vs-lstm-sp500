# Final Research Summary

## Project
*Comparaison des modèles statistiques et d'apprentissage automatique
pour la prévision financière (ARIMA-GARCH et LSTM)*

**Evaluation Period:** 2023-01-03 → 2026-08-13
**Evaluation Observations:** 906

### Table 1: Statistical Model Comparison
| Model                  |     AIC |     BIC |   Log-Likelihood |
|:-----------------------|--------:|--------:|-----------------:|
| Const+GJR-GARCH(1,1)+t | 12168.1 | 12206.7 |         -6078.05 |
| AR(1)+GJR-GARCH(1,1)+t | 12153.2 | 12198.2 |         -6069.58 |

### Table 2: LSTM Configurations
| Model              | Target                 | Features                               |   Lookback |   LSTM_Units |   Dropout |   Learning_Rate |   Batch_Size |   Max_Epochs |   Epochs_Trained |   Best_Val_Loss | Framework   | Optimizer   | Loss_Function   |   Seed |
|:-------------------|:-----------------------|:---------------------------------------|-----------:|-------------:|----------:|----------------:|-------------:|-------------:|-----------------:|----------------:|:------------|:------------|:----------------|-------:|
| LSTM_M1_Base       | Return                 | Log_Return                             |         60 |           32 |       0.2 |           0.001 |           64 |          100 |               58 |         2.02465 | PyTorch     | Adam        | MSE             |     42 |
| LSTM_M2_Enhanced   | Return                 | Log_Return + r^2 + RolVol21 + RolMAE21 |         60 |           32 |       0.2 |           0.001 |           64 |          100 |               22 |         2.07551 | PyTorch     | Adam        | MSE             |     42 |
| LSTM_M3_Volatility | r^2 (volatility proxy) | Log_Return + r^2 + RolVol21 + RolMAE21 |         60 |           32 |       0.2 |           0.001 |           64 |          100 |               23 |        54.173   | PyTorch     | Adam        | MSE             |     42 |

### Table 3: Return Forecasting Performance
| Model       |     RMSE |      MAE |   DirAcc(%) |   MAPE(%) |   QLIKE |
|:------------|---------:|---------:|------------:|----------:|--------:|
| GARCH_Const | 0.933423 | 0.655058 |     56.181  |       nan |     nan |
| GARCH_AR1   | 0.932899 | 0.656706 |     53.532  |       nan |     nan |
| LSTM_Base   | 0.930757 | 0.661675 |     49.4481 |       nan |     nan |
| LSTM_Enh    | 0.936954 | 0.664322 |     48.3444 |       nan |     nan |

### Table 4: Price Forecasting Performance
| Model       |    RMSE |     MAE |   DirAcc(%) |   MAPE(%) |   QLIKE |
|:------------|--------:|--------:|------------:|----------:|--------:|
| GARCH_Const | 52.1339 | 36.4456 |         nan |  0.654995 |     nan |
| GARCH_AR1   | 52.0898 | 36.522  |         nan |  0.656699 |     nan |
| LSTM_Base   | 51.9856 | 36.779  |         nan |  0.661486 |     nan |
| LSTM_Enh    | 52.3114 | 36.9289 |         nan |  0.664143 |     nan |

### Table 5: Volatility Forecasting Performance
| Model       |    RMSE |      MAE |   DirAcc(%) |   MAPE(%) |   QLIKE |
|:------------|--------:|---------:|------------:|----------:|--------:|
| GARCH_Const | 3.06621 | 0.946062 |         nan |       nan | 1.59364 |
| GARCH_AR1   | 3.05817 | 0.942743 |         nan |       nan | 1.59384 |
| LSTM_Vol    | 3.20398 | 1.00429  |         nan |       nan | 1.57726 |

### Table 6: Diebold-Mariano Tests
| Task       | Model_1     | Model_2   | Loss   |   DM_Stat |   p_value | Conclusion                                  |
|:-----------|:------------|:----------|:-------|----------:|----------:|:--------------------------------------------|
| Return     | GARCH_Const | LSTM_Base | SE     |    0.2144 |    0.8303 | No significant difference (p=0.8303)        |
| Return     | GARCH_Const | LSTM_Base | AE     |   -2.0087 |    0.0446 | GARCH_Const significantly better (p=0.0446) |
| Return     | GARCH_AR1   | LSTM_Enh  | SE     |   -0.548  |    0.5837 | No significant difference (p=0.5837)        |
| Return     | GARCH_AR1   | LSTM_Enh  | AE     |   -2.5392 |    0.0111 | GARCH_AR1 significantly better (p=0.0111)   |
| Return     | GARCH_Const | GARCH_AR1 | SE     |    0.1486 |    0.8819 | No significant difference (p=0.8819)        |
| Return     | LSTM_Base   | LSTM_Enh  | SE     |   -1.0631 |    0.2877 | No significant difference (p=0.2877)        |
| Volatility | GARCH_Const | LSTM_Vol  | QLIKE  |    0.7322 |    0.4641 | No significant difference (p=0.4641)        |
| Volatility | GARCH_Const | LSTM_Vol  | SE     |   -2.0313 |    0.0422 | GARCH_Const significantly better (p=0.0422) |
| Volatility | GARCH_AR1   | LSTM_Vol  | QLIKE  |    0.7305 |    0.4651 | No significant difference (p=0.4651)        |
| Volatility | GARCH_AR1   | LSTM_Vol  | SE     |   -2.0693 |    0.0385 | GARCH_AR1 significantly better (p=0.0385)   |

---

## Research Conclusions

### Summary
The above tables and figures provide the empirical evidence needed to answer
the core research question. Review the DM test p-values (Table 6) to determine
whether any observed metric differences are statistically significant at α=0.05.

### Methodological Notes
- All GARCH parameters estimated on Training set only (2000–2018)
- LSTM scalers fitted on Training set only
- No hyperparameter tuning on the Test set
- Chronological (non-shuffled) data split
- No outlier removal or data winsorization
- Reproducible random seed = 42

### Figures
| # | Description |
|---|-------------|
| 1 | Actual vs predicted returns |
| 2 | Actual vs predicted prices |
| 3 | Actual vs predicted volatility |
| 4 | Forecast errors over time |
| 5 | LSTM training/validation loss curves |
| 6 | Forecast accuracy during high-volatility periods |