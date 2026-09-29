# ARIMA-GARCH vs LSTM - Final Experiment Results
Generated: 2026-09-07 11:03

## 1. ARIMA Order Selection
Selected Order (BIC): ARIMA(2, 0, 0)
This was fit on the training data only.

## 2. Evaluation Metrics
| Task       | Model       |     RMSE |      MAE |   DirAcc |     QLIKE |
|:-----------|:------------|---------:|---------:|---------:|----------:|
| Return     | ARIMA-GARCH | 0.9387   | 0.665531 |  48.1236 | nan       |
| Return     | LSTM_Return | 0.930757 | 0.661675 |  49.4481 | nan       |
| Volatility | GARCH(1,1)  | 3.2465   | 1.02297  | nan      |   1.66164 |
| Volatility | LSTM_Vol    | 3.20398  | 1.00429  | nan      |   1.57726 |

## 3. Diebold-Mariano Tests
| Task       | Loss   |   DM_Stat |     p_value | Conclusion      |
|:-----------|:-------|----------:|------------:|:----------------|
| Return     | SE     |  0.744297 | 0.456697    | No sig diff     |
| Return     | AE     |  1.22498  | 0.220583    | No sig diff     |
| Volatility | QLIKE  |  4.25609  | 2.08033e-05 | LSTM_Vol better |
| Volatility | SE     |  0.536156 | 0.59185     | No sig diff     |

## 4. Methodological Notes
- 100% fixed parameters out-of-sample (no expanding window re-estimation).
- Two-step estimation used for ARIMA-GARCH.
- LSTMs retrained from scratch.
- No look-ahead bias, fully causal filters used.