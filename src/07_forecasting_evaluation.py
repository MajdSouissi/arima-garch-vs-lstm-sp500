"""
=============================================================================
07_forecasting_evaluation.py
Metrics: Returns (RMSE, MAE, DirAcc), Prices (RMSE, MAE, MAPE),
         Volatility (RMSE, MAE, QLIKE)
=============================================================================
Merges GARCH and LSTM out-of-sample forecasts on the common test dates
and computes all evaluation metrics.
=============================================================================
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent.parent)
GARCH_CSV   = os.path.join(BASE_DIR, "garch_forecasts.csv")
LSTM_CSV    = os.path.join(BASE_DIR, "lstm_forecasts.csv")
METRICS_CSV = os.path.join(BASE_DIR, "metrics_summary.csv")
REPORT_PATH = os.path.join(BASE_DIR, "07_evaluation_report.txt")

# â”€â”€ Metric Functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred)**2))

def mae(actual, pred):
    return np.mean(np.abs(actual - pred))

def directional_accuracy(actual, pred):
    """Percentage of days where sign(predicted) == sign(actual)."""
    return np.mean(np.sign(actual) == np.sign(pred)) * 100

def mape(actual, pred):
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100

def qlike(realized_var, forecast_var):
    """QLIKE loss: E[r^2/h - ln(r^2/h) - 1]."""
    eps = 1e-10
    rv = np.maximum(realized_var, eps)
    fv = np.maximum(forecast_var, eps)
    ratio = rv / fv
    return np.mean(ratio - np.log(ratio) - 1)

# â”€â”€ Load Data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Loading forecasts...")
garch_df = pd.read_csv(GARCH_CSV, parse_dates=['Date'])
lstm_df  = pd.read_csv(LSTM_CSV,  parse_dates=['Date'])

# GARCH test rows
garch_test = garch_df[garch_df['Split'] == 'Test'].copy()
print(f"GARCH test set: {len(garch_test)} rows")
print(f"  Range: {garch_test['Date'].min().date()} â†’ {garch_test['Date'].max().date()}")

print(f"LSTM test set:  {len(lstm_df)} rows")
print(f"  Range: {lstm_df['Date'].min()} â†’ {lstm_df['Date'].max()}")

# Merge on common dates (LSTM has fewer dates due to lookback window)
eval_df = pd.merge(garch_test, lstm_df, on='Date', how='inner', suffixes=('', '_lstm'))
print(f"Common evaluation dates: {len(eval_df)}")
print(f"  Range: {eval_df['Date'].min().date()} â†’ {eval_df['Date'].max().date()}")

if len(eval_df) == 0:
    print("ERROR: No overlapping dates between GARCH and LSTM forecasts!")
    exit(1)

# â”€â”€ Compute Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
metrics = []

# --- 1. Return Forecasting ---
actual_ret = eval_df['Realized_Return'].values

ret_models = {
    'GARCH_Const': eval_df['GARCH_M1_Mean'].values,
    'GARCH_AR1':   eval_df['GARCH_M2_Mean'].values,
    'LSTM_Base':   eval_df['LSTM_M1_Mean'].values,
    'LSTM_Enh':    eval_df['LSTM_M2_Mean'].values,
}

print("\n--- RETURN FORECASTING ---")
for name, pred in ret_models.items():
    mask = ~np.isnan(pred) & ~np.isnan(actual_ret)
    a, p = actual_ret[mask], pred[mask]
    r = rmse(a, p)
    m = mae(a, p)
    da = directional_accuracy(a, p)
    metrics.append({'Task': 'Return', 'Model': name, 'RMSE': r, 'MAE': m, 'DirAcc(%)': da})
    print(f"  {name:15s}  RMSE={r:.6f}  MAE={m:.6f}  DirAcc={da:.2f}%")

# --- 2. Price Forecasting ---
# P_hat_{t+1} = P_t * exp(r_hat/100) ... but our Log_Return is already 100*ln(P/P')
# So: P_hat = P_prev * exp(r_hat / 100)
# We need previous-day close prices
prev_close = eval_df['Close'].shift(1).values.copy()
actual_price = eval_df['Close'].values

# For the first row we need the close price from the day before
first_eval_date = eval_df['Date'].iloc[0]
pre_date_row = garch_df[garch_df['Date'] < first_eval_date].iloc[-1]
prev_close[0] = pre_date_row['Close']

print("\n--- PRICE FORECASTING ---")
for name, pred in ret_models.items():
    mask = ~np.isnan(pred) & ~np.isnan(prev_close) & ~np.isnan(actual_price)
    pred_price = prev_close[mask] * np.exp(pred[mask] / 100.0)
    a = actual_price[mask]
    r = rmse(a, pred_price)
    m = mae(a, pred_price)
    mp = mape(a, pred_price)
    metrics.append({'Task': 'Price', 'Model': name, 'RMSE': r, 'MAE': m, 'MAPE(%)': mp})
    print(f"  {name:15s}  RMSE={r:.4f}  MAE={m:.4f}  MAPE={mp:.4f}%")

# --- 3. Volatility Evaluation ---
vol_actual = eval_df['Realized_Vol_Proxy'].values # this is r^2_t

vol_models = {
    'GARCH_Const': eval_df['GARCH_M1_Vol'].values**2,  # GARCH outputs sigma, square to get var
    'GARCH_AR1':   eval_df['GARCH_M2_Vol'].values**2,  # GARCH outputs sigma, square to get var
    'LSTM_Vol':    eval_df['LSTM_M3_Var'].values,      # LSTM M3 directly outputs var
}

print("\n--- VOLATILITY FORECASTING ---")
for name, pred_var in vol_models.items():
    mask = ~np.isnan(pred_var) & ~np.isnan(vol_actual)
    rv, fv = vol_actual[mask], pred_var[mask]
    r = rmse(rv, fv)
    m = mae(rv, fv)
    q = qlike(rv, fv)
    metrics.append({'Task': 'Volatility', 'Model': name, 'RMSE': r, 'MAE': m, 'QLIKE': q})
    print(f"  {name:15s}  RMSE={r:.6f}  MAE={m:.6f}  QLIKE={q:.6f}")

# â”€â”€ Save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(METRICS_CSV, index=False)
print(f"\nMetrics saved to: {METRICS_CSV}")

# Also save the merged evaluation dataframe for downstream scripts
eval_df.to_csv(os.path.join(BASE_DIR, 'evaluation_merged.csv'), index=False)
print(f"Merged evaluation data saved to: evaluation_merged.csv")

# Text report
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write("FORECASTING EVALUATION REPORT\n")
    f.write(f"Evaluation period: {eval_df['Date'].min().date()} to {eval_df['Date'].max().date()}\n")
    f.write(f"Number of evaluation days: {len(eval_df)}\n\n")
    f.write(metrics_df.to_string(index=False))
print(f"Report saved to: {REPORT_PATH}")
print("\nâœ… Evaluation complete.")
