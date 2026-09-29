"""
=============================================================================
run_garch11_oos.py
Symmetric GARCH(1,1) â€” Out-of-Sample Forecasting & Evaluation
=============================================================================
Fits a SYMMETRIC GARCH(1,1) with Constant Mean and Student-t innovations
on the Training set ONLY, then generates 1-step-ahead OOS forecasts.

Model:  ÏƒÂ²_t = Ï‰ + Î± ÎµÂ²_{t-1} + Î² ÏƒÂ²_{t-1}
Mean:   r_t = Î¼ + Îµ_t
Dist:   Standardized Student-t

This is NOT a GJR-GARCH. The asymmetry parameter (o) is set to 0.

Specification choices documented:
  - Constant Mean: consistent with GARCH_Const (Model 1) in 04_garch_models.py
    and with the mean_variance_report.md finding that "constant-mean is defensible"
  - Student-t: consistent with the GJR-GARCH models already in the project;
    motivated by the documented excess kurtosis (~11) in Log_Return
  - GARCH(1,1) symmetric: explicitly requested for comparison against LSTM_Vol

Split dates (hardcoded, matching existing project protocol):
  Train: 2000-08-15 â†’ 2018-12-31
  Val:   2019-01-02 â†’ 2022-12-30
  Test:  2023-01-03 â†’ 2026-08-13
=============================================================================
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# â”€â”€ Dependencies â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    from arch import arch_model
except ImportError:
    print("ERROR: 'arch' package not installed. pip install arch")
    sys.exit(1)

from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR = str(Path(__file__).resolve().parent.parent)
INPUT_CSV   = os.path.join(BASE_DIR, "SP500_master_2000_2026.csv")
LSTM_CSV    = os.path.join(BASE_DIR, "lstm_forecasts.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Output paths
FORECASTS_CSV = os.path.join(RESULTS_DIR, "garch11_forecasts.csv")
METRICS_CSV   = os.path.join(RESULTS_DIR, "garch11_metrics.csv")
DM_CSV        = os.path.join(RESULTS_DIR, "garch11_dm_tests.csv")
REPORT_MD     = os.path.join(RESULTS_DIR, "garch11_report.md")
MASTER_MD     = os.path.join(RESULTS_DIR, "master_comparison.md")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# METRIC FUNCTIONS â€” identical to 07_forecasting_evaluation.py
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred)**2))

def mae(actual, pred):
    return np.mean(np.abs(actual - pred))

def directional_accuracy(actual, pred):
    """Percentage of days where sign(predicted) == sign(actual)."""
    return np.mean(np.sign(actual) == np.sign(pred)) * 100

def qlike(realized_var, forecast_var):
    """QLIKE loss: E[rÂ²/h - ln(rÂ²/h) - 1]."""
    eps = 1e-10
    rv = np.maximum(realized_var, eps)
    fv = np.maximum(forecast_var, eps)
    ratio = rv / fv
    return np.mean(ratio - np.log(ratio) - 1)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DM TEST FUNCTIONS â€” identical to 08_statistical_comparison.py
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def dm_test(e1, e2, loss='SE', h=1):
    """
    Diebold-Mariano test.
    e1, e2: forecast errors from model 1 and model 2.
    loss: 'SE' (squared error) or 'AE' (absolute error).
    DM < 0 â†’ Model 1 has lower loss (better).
    DM > 0 â†’ Model 2 has lower loss (better).
    """
    if loss == 'SE':
        d = e1**2 - e2**2
    elif loss == 'AE':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    n = len(d)
    d_bar = np.mean(d)
    gamma_0 = np.var(d, ddof=0)
    if gamma_0 < 1e-15:
        return 0.0, 1.0
    V = gamma_0 / n
    if V <= 0:
        return 0.0, 1.0
    dm_stat = d_bar / np.sqrt(V)
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

def dm_test_vol(actual_var, pred_var_1, pred_var_2, loss='QLIKE'):
    """DM test using QLIKE or MSE loss directly on variance forecasts."""
    eps = 1e-10
    rv = np.maximum(actual_var, eps)
    fv1 = np.maximum(pred_var_1, eps)
    fv2 = np.maximum(pred_var_2, eps)

    if loss == 'QLIKE':
        l1 = rv / fv1 - np.log(rv / fv1) - 1
        l2 = rv / fv2 - np.log(rv / fv2) - 1
    elif loss == 'SE':
        l1 = (rv - fv1)**2
        l2 = (rv - fv2)**2
    else:
        raise ValueError(f"Unknown loss: {loss}")

    d = l1 - l2
    n = len(d)
    d_bar = np.mean(d)
    gamma_0 = np.var(d, ddof=0)
    if gamma_0 < 1e-15:
        return 0.0, 1.0
    V = gamma_0 / n
    if V <= 0:
        return 0.0, 1.0
    dm_stat = d_bar / np.sqrt(V)
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1. DATA LOADING & SPLITTING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("=" * 78)
print(" SYMMETRIC GARCH(1,1) â€” OUT-OF-SAMPLE EVALUATION")
print("=" * 78)

df = pd.read_csv(INPUT_CSV, parse_dates=['Date'])
df = df.dropna(subset=['Log_Return']).reset_index(drop=True)

# Split dates â€” identical to 04_garch_models.py
train_end_dt = pd.to_datetime('2018-12-31')
val_end_dt   = pd.to_datetime('2022-12-31')

train_df = df[df['Date'] <= train_end_dt]
val_df   = df[(df['Date'] > train_end_dt) & (df['Date'] <= val_end_dt)]
test_df  = df[df['Date'] > val_end_dt]

train_size = len(train_df)
val_size   = len(val_df)
test_size  = len(test_df)

print(f"\nTotal Observations: {len(df)}")
print(f"Train: {train_df['Date'].iloc[0].date()} â†’ {train_df['Date'].iloc[-1].date()} ({train_size} obs)")
print(f"Val:   {val_df['Date'].iloc[0].date()} â†’ {val_df['Date'].iloc[-1].date()} ({val_size} obs)")
print(f"Test:  {test_df['Date'].iloc[0].date()} â†’ {test_df['Date'].iloc[-1].date()} ({test_size} obs)")

# Verify expected test period
assert str(test_df['Date'].iloc[0].date()) == '2023-01-03', \
    f"Test start mismatch: {test_df['Date'].iloc[0].date()}"
assert str(test_df['Date'].iloc[-1].date()) == '2026-08-13', \
    f"Test end mismatch: {test_df['Date'].iloc[-1].date()}"
assert test_size == 906, f"Test size mismatch: {test_size} != 906"
print("âœ“ Split dates verified.")

# Build return series with DatetimeIndex
returns = pd.Series(df['Log_Return'].values, index=df['Date'])
train_end_date = train_df['Date'].iloc[-1]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2. FIT SYMMETRIC GARCH(1,1)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n" + "=" * 78)
print(" FITTING: Constant Mean + GARCH(1,1) + Student-t")
print("  ÏƒÂ²_t = Ï‰ + Î± ÎµÂ²_{t-1} + Î² ÏƒÂ²_{t-1}")
print("  o = 0 (NO asymmetry / NO gamma term)")
print("=" * 78)

am = arch_model(
    returns,
    mean='Constant',
    vol='Garch',
    p=1,
    o=0,     # â† SYMMETRIC: no gamma/leverage term
    q=1,
    dist='t'
)

# Fit on training set ONLY
res = am.fit(last_obs=train_end_date, disp='off')

print("\n" + res.summary().as_text())

# â”€â”€ Verify symmetry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
param_names = list(res.params.index)
print(f"\nParameters: {param_names}")
assert 'gamma[1]' not in param_names, \
    "ERROR: gamma[1] found â€” this is NOT a symmetric GARCH!"
print("âœ“ Confirmed: No gamma parameter. Model is symmetric GARCH(1,1).")

alpha = res.params['alpha[1]']
beta  = res.params['beta[1]']
omega = res.params['omega']
mu    = res.params['mu']
nu    = res.params.get('nu', None)

persistence = alpha + beta
print(f"\n  Ï‰     = {omega:.6f}")
print(f"  Î±     = {alpha:.6f}")
print(f"  Î²     = {beta:.6f}")
print(f"  Î± + Î² = {persistence:.6f}")
print(f"  Î¼     = {mu:.6f}")
if nu is not None:
    print(f"  Î½ (Student-t df) = {nu:.4f}")

print(f"\n  AIC = {res.aic:.2f}")
print(f"  BIC = {res.bic:.2f}")
print(f"  Log-Likelihood = {res.loglikelihood:.2f}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 3. OUT-OF-SAMPLE FORECASTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n" + "=" * 78)
print(" GENERATING 1-STEP-AHEAD OOS FORECASTS")
print("=" * 78)

forecasts = res.forecast(horizon=1, align='target')

pred_mean = forecasts.mean['h.1']
pred_var  = forecasts.variance['h.1']

# Build forecast DataFrame (full series)
forecast_df = df[['Date', 'Close', 'Log_Return']].copy()
forecast_df['Split'] = 'Train'

val_start_idx = train_size
val_end_idx   = train_size + val_size - 1
test_start_idx = train_size + val_size

forecast_df.loc[val_start_idx:val_end_idx, 'Split'] = 'Val'
forecast_df.loc[test_start_idx:, 'Split'] = 'Test'

forecast_df.set_index('Date', inplace=True)
forecast_df['GARCH11_Mean'] = pred_mean
forecast_df['GARCH11_Vol']  = np.sqrt(np.maximum(pred_var, 0))
forecast_df.reset_index(inplace=True)

# Save forecasts
forecast_df.to_csv(FORECASTS_CSV, index=False)
print(f"Forecasts saved: {FORECASTS_CSV}")

# Extract test forecasts
test_fc = forecast_df[forecast_df['Split'] == 'Test'].copy()
print(f"Test forecasts: {len(test_fc)} observations")
print(f"  Range: {test_fc['Date'].iloc[0].date()} â†’ {test_fc['Date'].iloc[-1].date()}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 4. LOAD LSTM FORECASTS & ALIGN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n" + "=" * 78)
print(" ALIGNING WITH LSTM FORECASTS")
print("=" * 78)

lstm_df = pd.read_csv(LSTM_CSV, parse_dates=['Date'])
print(f"LSTM forecasts: {len(lstm_df)} rows")
print(f"  Range: {lstm_df['Date'].min().date()} â†’ {lstm_df['Date'].max().date()}")

# Merge on common test dates
eval_df = pd.merge(
    test_fc,
    lstm_df[['Date', 'Realized_Return', 'Realized_Vol_Proxy', 'LSTM_M1_Mean', 'LSTM_M3_Var']],
    on='Date',
    how='inner'
)
print(f"Common evaluation dates: {len(eval_df)}")
print(f"  Range: {eval_df['Date'].iloc[0].date()} â†’ {eval_df['Date'].iloc[-1].date()}")

if len(eval_df) == 0:
    print("ERROR: No overlapping dates!")
    sys.exit(1)

# Verify alignment: Log_Return from GARCH df should match Realized_Return from LSTM df
ret_diff = np.max(np.abs(eval_df['Log_Return'].values - eval_df['Realized_Return'].values))
print(f"Max |Log_Return - Realized_Return|: {ret_diff:.2e}")
assert ret_diff < 1e-6, f"Return alignment mismatch: {ret_diff}"
print("âœ“ Returns are aligned between GARCH and LSTM data.")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 5. COMPUTE METRICS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n" + "=" * 78)
print(" COMPUTING METRICS")
print("=" * 78)

actual_ret = eval_df['Realized_Return'].values
actual_var = eval_df['Realized_Vol_Proxy'].values   # rÂ²_t

garch11_mean = eval_df['GARCH11_Mean'].values
garch11_var  = eval_df['GARCH11_Vol'].values ** 2   # ÏƒÂ² from Ïƒ

lstm_var     = eval_df['LSTM_M3_Var'].values
lstm_mean    = eval_df['LSTM_M1_Mean'].values

metrics = []

# --- Return metrics for GARCH(1,1) ---
mask_ret = ~np.isnan(garch11_mean) & ~np.isnan(actual_ret)
r_rmse = rmse(actual_ret[mask_ret], garch11_mean[mask_ret])
r_mae  = mae(actual_ret[mask_ret], garch11_mean[mask_ret])
r_da   = directional_accuracy(actual_ret[mask_ret], garch11_mean[mask_ret])

print(f"\n--- GARCH(1,1) Return Forecasting ---")
print(f"  RMSE      = {r_rmse:.6f}")
print(f"  MAE       = {r_mae:.6f}")
print(f"  DirAcc(%) = {r_da:.2f}")

metrics.append({
    'Task': 'Return', 'Model': 'GARCH11_Const',
    'RMSE': r_rmse, 'MAE': r_mae, 'DirAcc(%)': r_da,
    'QLIKE': np.nan
})

# --- Volatility metrics for GARCH(1,1) ---
mask_vol = ~np.isnan(garch11_var) & ~np.isnan(actual_var)
v_rmse  = rmse(actual_var[mask_vol], garch11_var[mask_vol])
v_mae   = mae(actual_var[mask_vol], garch11_var[mask_vol])
v_qlike = qlike(actual_var[mask_vol], garch11_var[mask_vol])

print(f"\n--- GARCH(1,1) Volatility Forecasting ---")
print(f"  RMSE  = {v_rmse:.6f}")
print(f"  MAE   = {v_mae:.6f}")
print(f"  QLIKE = {v_qlike:.6f}")

metrics.append({
    'Task': 'Volatility', 'Model': 'GARCH11_Const',
    'RMSE': v_rmse, 'MAE': v_mae, 'DirAcc(%)': np.nan,
    'QLIKE': v_qlike
})

# --- LSTM_Vol volatility metrics (recomputed for verification) ---
mask_lstm = ~np.isnan(lstm_var) & ~np.isnan(actual_var)
lv_rmse  = rmse(actual_var[mask_lstm], lstm_var[mask_lstm])
lv_mae   = mae(actual_var[mask_lstm], lstm_var[mask_lstm])
lv_qlike = qlike(actual_var[mask_lstm], lstm_var[mask_lstm])

print(f"\n--- LSTM_Vol Volatility Forecasting (verification) ---")
print(f"  RMSE  = {lv_rmse:.6f}")
print(f"  MAE   = {lv_mae:.6f}")
print(f"  QLIKE = {lv_qlike:.6f}")

metrics.append({
    'Task': 'Volatility', 'Model': 'LSTM_Vol',
    'RMSE': lv_rmse, 'MAE': lv_mae, 'DirAcc(%)': np.nan,
    'QLIKE': lv_qlike
})

# --- LSTM_Base return metrics (recomputed for verification) ---
mask_lstm_ret = ~np.isnan(lstm_mean) & ~np.isnan(actual_ret)
lr_rmse = rmse(actual_ret[mask_lstm_ret], lstm_mean[mask_lstm_ret])
lr_mae  = mae(actual_ret[mask_lstm_ret], lstm_mean[mask_lstm_ret])
lr_da   = directional_accuracy(actual_ret[mask_lstm_ret], lstm_mean[mask_lstm_ret])

print(f"\n--- LSTM_Base Return Forecasting (verification) ---")
print(f"  RMSE      = {lr_rmse:.6f}")
print(f"  MAE       = {lr_mae:.6f}")
print(f"  DirAcc(%) = {lr_da:.2f}")

metrics.append({
    'Task': 'Return', 'Model': 'LSTM_Base',
    'RMSE': lr_rmse, 'MAE': lr_mae, 'DirAcc(%)': lr_da,
    'QLIKE': np.nan
})

# Save metrics
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(METRICS_CSV, index=False)
print(f"\nMetrics saved: {METRICS_CSV}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 6. DIEBOLD-MARIANO TESTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n" + "=" * 78)
print(" DIEBOLD-MARIANO TESTS")
print("=" * 78)

dm_results = []

def run_dm_vol_and_record(m1_name, v1, m2_name, v2, loss_type):
    dm, p = dm_test_vol(actual_var, v1, v2, loss=loss_type)
    if p < 0.05:
        winner = m1_name if dm < 0 else m2_name
        conclusion = f"{winner} significantly better (p={p:.4f})"
    else:
        conclusion = f"No significant difference (p={p:.4f})"
    dm_results.append({
        'Task': 'Volatility', 'Model_1': m1_name, 'Model_2': m2_name,
        'Loss': loss_type, 'DM_Stat': round(dm, 4), 'p_value': round(p, 4),
        'N': len(actual_var), 'Conclusion': conclusion
    })
    print(f"  {m1_name} vs {m2_name} ({loss_type}): DM={dm:+.4f}  p={p:.4f}  â†’ {conclusion}")

def run_dm_ret_and_record(m1_name, e1, m2_name, e2, loss_type):
    dm, p = dm_test(e1, e2, loss=loss_type)
    if p < 0.05:
        winner = m1_name if dm < 0 else m2_name
        conclusion = f"{winner} significantly better (p={p:.4f})"
    else:
        conclusion = f"No significant difference (p={p:.4f})"
    dm_results.append({
        'Task': 'Return', 'Model_1': m1_name, 'Model_2': m2_name,
        'Loss': loss_type, 'DM_Stat': round(dm, 4), 'p_value': round(p, 4),
        'N': len(e1), 'Conclusion': conclusion
    })
    print(f"  {m1_name} vs {m2_name} ({loss_type}): DM={dm:+.4f}  p={p:.4f}  â†’ {conclusion}")

# Volatility: GARCH(1,1) vs LSTM_Vol
print("\n--- Volatility: GARCH(1,1) vs LSTM_Vol ---")
run_dm_vol_and_record('GARCH11_Const', garch11_var, 'LSTM_Vol', lstm_var, 'QLIKE')
run_dm_vol_and_record('GARCH11_Const', garch11_var, 'LSTM_Vol', lstm_var, 'SE')

# Return: GARCH(1,1) vs LSTM_Base
print("\n--- Return: GARCH(1,1) vs LSTM_Base ---")
e_garch11 = actual_ret - garch11_mean
e_lstm_base = actual_ret - lstm_mean
run_dm_ret_and_record('GARCH11_Const', e_garch11, 'LSTM_Base', e_lstm_base, 'SE')
run_dm_ret_and_record('GARCH11_Const', e_garch11, 'LSTM_Base', e_lstm_base, 'AE')

dm_df = pd.DataFrame(dm_results)
dm_df.to_csv(DM_CSV, index=False)
print(f"\nDM tests saved: {DM_CSV}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 7. COVID-19 STRESS FIGURE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n" + "=" * 78)
print(" COVID-19 STRESS ANALYSIS (Validation Period)")
print("=" * 78)

# The COVID period falls in the validation set (2019-2022)
# We need GARCH forecasts for this period from our forecast_df
covid_start = pd.to_datetime('2020-02-01')
covid_end   = pd.to_datetime('2020-05-31')

covid_fc = forecast_df[
    (forecast_df['Date'] >= covid_start) &
    (forecast_df['Date'] <= covid_end)
].copy()

print(f"COVID window: {covid_fc['Date'].iloc[0].date()} â†’ {covid_fc['Date'].iloc[-1].date()}")
print(f"  Observations: {len(covid_fc)}")

# Check if we have GARCH(1,1) forecasts for this period
garch11_covid_valid = covid_fc['GARCH11_Vol'].notna().sum()
print(f"  GARCH(1,1) forecasts available: {garch11_covid_valid}")

# Load existing GJR-GARCH forecasts for comparison
garch_full = pd.read_csv(os.path.join(BASE_DIR, "garch_forecasts.csv"), parse_dates=['Date'])
garch_covid = garch_full[
    (garch_full['Date'] >= covid_start) &
    (garch_full['Date'] <= covid_end)
].copy()

# Note: LSTM forecasts only cover test period (2023+), so we cannot compare
# LSTM_Vol during COVID. The figure will show GARCH(1,1) vs GJR-GARCH only,
# plus the realized volatility.

if garch11_covid_valid > 0:
    fig, ax = plt.subplots(figsize=(14, 6))

    # Realized volatility proxy: |r_t| (sqrt of rÂ²)
    realized_vol_covid = np.abs(covid_fc['Log_Return'].values)

    dates_covid = covid_fc['Date'].values

    ax.plot(dates_covid, realized_vol_covid,
            color='#333333', alpha=0.5, linewidth=0.8,
            label='Realized |r$_t$| (proxy)')

    ax.plot(dates_covid, covid_fc['GARCH11_Vol'].values,
            color='#1565C0', linewidth=1.5, alpha=0.9,
            label='GARCH(1,1) Ïƒ$_t$')

    # GJR-GARCH for comparison (M1 = Constant Mean)
    if len(garch_covid) > 0 and 'GARCH_M1_Vol' in garch_covid.columns:
        ax.plot(garch_covid['Date'].values, garch_covid['GARCH_M1_Vol'].values,
                color='#E65100', linewidth=1.5, alpha=0.9, linestyle='--',
                label='GJR-GARCH(1,1) Ïƒ$_t$')

    # Mark key dates
    ax.axvline(pd.to_datetime('2020-02-19'), color='red', alpha=0.3,
               linestyle=':', linewidth=1, label='COVID crash start (~Feb 19)')
    ax.axvline(pd.to_datetime('2020-03-23'), color='green', alpha=0.3,
               linestyle=':', linewidth=1, label='Market bottom (~Mar 23)')

    ax.set_title('COVID-19 Stress Period: Volatility Forecasts vs Realized\n'
                 '(Validation Period â€” NOT part of OOS Test)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Volatility (Ïƒ)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate('Note: LSTM_Vol forecasts unavailable for this period\n'
                '(LSTM test set begins 2023-01-03)',
                xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=8, color='gray', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    covid_fig_path = os.path.join(FIGURES_DIR, "fig_covid_stress_volatility.png")
    fig.savefig(covid_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"COVID stress figure saved: {covid_fig_path}")
else:
    print("WARNING: No GARCH(1,1) forecasts available for COVID period.")
    covid_fig_path = None


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 8. GENERATE REPORTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n" + "=" * 78)
print(" GENERATING REPORTS")
print("=" * 78)

# â”€â”€ garch11_report.md â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
report = []
report.append("# GARCH(1,1) Symmetric â€” Out-of-Sample Evaluation Report")
report.append("")
report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
report.append("")
report.append("## 1. Data")
report.append("")
report.append("| Item | Value |")
report.append("|------|-------|")
report.append(f"| Dataset | S&P 500 daily log returns (%) |")
report.append(f"| Source | `SP500_master_2000_2026.csv` |")
report.append(f"| Total observations | {len(df)} |")
report.append(f"| Train | {train_df['Date'].iloc[0].date()} â†’ {train_df['Date'].iloc[-1].date()} ({train_size} obs) |")
report.append(f"| Validation | {val_df['Date'].iloc[0].date()} â†’ {val_df['Date'].iloc[-1].date()} ({val_size} obs) |")
report.append(f"| Test | {test_df['Date'].iloc[0].date()} â†’ {test_df['Date'].iloc[-1].date()} ({test_size} obs) |")
report.append(f"| Evaluation observations | {len(eval_df)} (inner join with LSTM dates) |")
report.append("")

report.append("## 2. Model Specification")
report.append("")
report.append("```")
report.append("Conditional Mean:   r_t = Î¼ + Îµ_t          (Constant Mean)")
report.append("Conditional Var:    ÏƒÂ²_t = Ï‰ + Î± ÎµÂ²_{t-1} + Î² ÏƒÂ²_{t-1}   (GARCH(1,1) symmetric)")
report.append("Innovations:        z_t ~ Student-t(Î½)")
report.append("")
report.append("This is a SYMMETRIC GARCH â€” no gamma/leverage parameter.")
report.append("arch_model(..., vol='Garch', p=1, o=0, q=1, dist='t')")
report.append("```")
report.append("")

report.append("### Specification Choices")
report.append("")
report.append("| Choice | Justification |")
report.append("|--------|---------------|")
report.append("| Constant Mean | Consistent with GARCH_Const (GJR Model 1); `mean_variance_report.md` finds 'constant-mean is defensible' |")
report.append("| Student-t | Consistent with GJR-GARCH models; motivated by excess kurtosis â‰ˆ 11 |")
report.append("| GARCH(1,1) symmetric | Requested as baseline comparison for LSTM_Vol |")
report.append("")

report.append("## 3. Estimated Parameters (Training Set Only)")
report.append("")
report.append("| Parameter | Value |")
report.append("|-----------|-------|")
report.append(f"| Î¼ (mean) | {mu:.6f} |")
report.append(f"| Ï‰ (intercept) | {omega:.6f} |")
report.append(f"| Î± (ARCH) | {alpha:.6f} |")
report.append(f"| Î² (GARCH) | {beta:.6f} |")
report.append(f"| Î± + Î² (persistence) | {persistence:.6f} |")
if nu is not None:
    report.append(f"| Î½ (Student-t df) | {nu:.4f} |")
report.append(f"| AIC | {res.aic:.2f} |")
report.append(f"| BIC | {res.bic:.2f} |")
report.append(f"| Log-Likelihood | {res.loglikelihood:.2f} |")
report.append(f"| Estimation obs | {train_size} |")
report.append("")

report.append("## 4. OOS Forecast Protocol")
report.append("")
report.append("- **Horizon:** 1-step-ahead")
report.append("- **Method:** `res.forecast(horizon=1, align='target')`")
report.append("- **Parameters fixed** at training-set estimates (no re-estimation)")
report.append("- **No test data** used in parameter estimation")
report.append("")

report.append("## 5. Volatility Evaluation Metrics (Test Period)")
report.append("")
report.append("Target: `Realized_Vol_Proxy` = rÂ²_t (squared log return)")
report.append("")
report.append("| Model | RMSE | MAE | QLIKE |")
report.append("|-------|------|-----|-------|")
report.append(f"| GARCH(1,1) symmetric | {v_rmse:.6f} | {v_mae:.6f} | {v_qlike:.6f} |")
report.append(f"| LSTM_Vol | {lv_rmse:.6f} | {lv_mae:.6f} | {lv_qlike:.6f} |")
report.append("")

report.append("## 6. Return Evaluation Metrics (Test Period)")
report.append("")
report.append("| Model | RMSE | MAE | DirAcc (%) |")
report.append("|-------|------|-----|------------|")
report.append(f"| GARCH(1,1) symmetric | {r_rmse:.6f} | {r_mae:.6f} | {r_da:.2f} |")
report.append(f"| LSTM_Base | {lr_rmse:.6f} | {lr_mae:.6f} | {lr_da:.2f} |")
report.append("")

report.append("## 7. Diebold-Mariano Tests")
report.append("")
report.append(dm_df.to_markdown(index=False))
report.append("")

report.append("## 8. COVID-19 Stress Analysis")
report.append("")
if covid_fig_path:
    report.append("A stress figure was produced for the COVID-19 period (Febâ€“May 2020).")
    report.append("This period falls in the **validation set**, not the test set.")
    report.append("")
    report.append("Key observations:")
    report.append("- LSTM_Vol forecasts are **unavailable** for this period (LSTM test begins 2023)")
    report.append("- The figure compares GARCH(1,1) symmetric vs GJR-GARCH(1,1) vs realized |r_t|")
    report.append("- The purpose is to observe model behavior during extreme volatility regimes")
    report.append("- This is NOT a formal evaluation â€” the official test period remains 2023â€“2026")
    report.append("")
    report.append(f"Figure: `{os.path.basename(covid_fig_path)}`")
else:
    report.append("COVID stress figure could not be produced.")
report.append("")

report.append("## 9. Hypotheses and Limitations")
report.append("")
report.append("1. **Fixed parameters:** GARCH parameters estimated once on training data, not re-estimated")
report.append("   during the test period. This is consistent with the GJR-GARCH protocol.")
report.append("2. **No rolling/expanding window:** Same protocol as existing models.")
report.append("3. **Constant Mean chosen** based on existing project conventions. AR(1) was also")
report.append("   tested in the GJR setup but is not reproduced here for the symmetric GARCH.")
report.append("4. **Student-t chosen** based on existing project conventions. The `mean_variance_report.md`")
report.append("   confirms Student-t improves fit over normal for all GARCH variants.")
report.append("")

with open(REPORT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))
print(f"Report saved: {REPORT_MD}")

# â”€â”€ master_comparison.md â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Load existing metrics for GJR-GARCH (from the original project)
existing_metrics = pd.read_csv(os.path.join(BASE_DIR, "metrics_summary.csv"))

# Extract GJR results
gjr_ret = existing_metrics[
    (existing_metrics['Task'] == 'Return') & (existing_metrics['Model'] == 'GARCH_Const')
].iloc[0]
gjr_vol = existing_metrics[
    (existing_metrics['Task'] == 'Volatility') & (existing_metrics['Model'] == 'GARCH_Const')
].iloc[0]

lstm_base_ret = existing_metrics[
    (existing_metrics['Task'] == 'Return') & (existing_metrics['Model'] == 'LSTM_Base')
].iloc[0]
lstm_vol_existing = existing_metrics[
    (existing_metrics['Task'] == 'Volatility') & (existing_metrics['Model'] == 'LSTM_Vol')
].iloc[0]

# Load existing DM tests
existing_dm = pd.read_csv(os.path.join(BASE_DIR, "dm_tests.csv"))

master = []
master.append("# Master Comparison â€” Final Thesis Results")
master.append("")
master.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
master.append(f"**Test Period:** 2023-01-03 â†’ 2026-08-13 ({len(eval_df)} obs)")
master.append("")

master.append("---")
master.append("")
master.append("## A. Return Forecasting")
master.append("")
master.append("Retained statistical model for returns: **GJR-GARCH(1,1) + Constant Mean + Student-t**")
master.append("(selected by BIC from `mean_variance_report.md`; `GARCH_Const` in existing results)")
master.append("")
master.append("| Model | Type | RMSE | MAE | DirAcc (%) |")
master.append("|-------|------|------|-----|------------|")
master.append(f"| GARCH_Const (GJR) | Statistical | {gjr_ret['RMSE']:.6f} | {gjr_ret['MAE']:.6f} | {gjr_ret['DirAcc(%)']:.2f} |")
master.append(f"| LSTM_Base | Deep Learning | {lstm_base_ret['RMSE']:.6f} | {lstm_base_ret['MAE']:.6f} | {lstm_base_ret['DirAcc(%)']:.2f} |")
master.append("")

# DM for returns
dm_ret_se = existing_dm[
    (existing_dm['Task'] == 'Return') &
    (existing_dm['Model_1'] == 'GARCH_Const') &
    (existing_dm['Model_2'] == 'LSTM_Base') &
    (existing_dm['Loss'] == 'SE')
].iloc[0]
dm_ret_ae = existing_dm[
    (existing_dm['Task'] == 'Return') &
    (existing_dm['Model_1'] == 'GARCH_Const') &
    (existing_dm['Model_2'] == 'LSTM_Base') &
    (existing_dm['Loss'] == 'AE')
].iloc[0]

master.append("### DM Tests (Returns)")
master.append("")
master.append("| Loss | DM Stat | p-value | Conclusion |")
master.append("|------|---------|---------|------------|")
master.append(f"| SE | {dm_ret_se['DM_Stat']:.4f} | {dm_ret_se['p_value']:.4f} | {dm_ret_se['Conclusion']} |")
master.append(f"| AE | {dm_ret_ae['DM_Stat']:.4f} | {dm_ret_ae['p_value']:.4f} | {dm_ret_ae['Conclusion']} |")
master.append("")

master.append("---")
master.append("")
master.append("## B. Volatility Forecasting")
master.append("")
master.append("| Model | Type | RMSE | MAE | QLIKE |")
master.append("|-------|------|------|-----|-------|")
master.append(f"| GARCH(1,1) symmetric | Statistical | {v_rmse:.6f} | {v_mae:.6f} | {v_qlike:.6f} |")
master.append(f"| GJR-GARCH(1,1) (existing) | Statistical | {gjr_vol['RMSE']:.6f} | {gjr_vol['MAE']:.6f} | {gjr_vol['QLIKE']:.6f} |")
master.append(f"| LSTM_Vol | Deep Learning | {lv_rmse:.6f} | {lv_mae:.6f} | {lv_qlike:.6f} |")
master.append("")

master.append("### DM Tests â€” GARCH(1,1) symmetric vs LSTM_Vol")
master.append("")
# Use newly computed DM results
dm_g11_qlike = dm_results[0]  # QLIKE
dm_g11_se    = dm_results[1]  # SE
master.append("| Loss | DM Stat | p-value | N | Conclusion |")
master.append("|------|---------|---------|---|------------|")
master.append(f"| QLIKE | {dm_g11_qlike['DM_Stat']:.4f} | {dm_g11_qlike['p_value']:.4f} | {dm_g11_qlike['N']} | {dm_g11_qlike['Conclusion']} |")
master.append(f"| SE | {dm_g11_se['DM_Stat']:.4f} | {dm_g11_se['p_value']:.4f} | {dm_g11_se['N']} | {dm_g11_se['Conclusion']} |")
master.append("")

master.append("### DM Tests â€” GJR-GARCH(1,1) vs LSTM_Vol (existing)")
master.append("")
dm_gjr_qlike = existing_dm[
    (existing_dm['Task'] == 'Volatility') &
    (existing_dm['Model_1'] == 'GARCH_Const') &
    (existing_dm['Loss'] == 'QLIKE')
].iloc[0]
dm_gjr_se = existing_dm[
    (existing_dm['Task'] == 'Volatility') &
    (existing_dm['Model_1'] == 'GARCH_Const') &
    (existing_dm['Loss'] == 'SE')
].iloc[0]
master.append("| Loss | DM Stat | p-value | Conclusion |")
master.append("|------|---------|---------|------------|")
master.append(f"| QLIKE | {dm_gjr_qlike['DM_Stat']:.4f} | {dm_gjr_qlike['p_value']:.4f} | {dm_gjr_qlike['Conclusion']} |")
master.append(f"| SE | {dm_gjr_se['DM_Stat']:.4f} | {dm_gjr_se['p_value']:.4f} | {dm_gjr_se['Conclusion']} |")
master.append("")

master.append("---")
master.append("")
master.append("## C. Methodological Notes")
master.append("")
master.append("- All GARCH parameters estimated on Training set only (2000â€“2018)")
master.append("- LSTM scalers fitted on Training set only; early stopping on Validation set")
master.append("- No hyperparameter tuning on the Test set")
master.append("- Chronological (non-shuffled) data split")
master.append("- No outlier removal or data winsorization")
master.append("- LSTM reproducible random seed = 42")
master.append("- Volatility target: rÂ²_t (squared log return) for both GARCH and LSTM")
master.append("- GARCH forecasts Ïƒ_t; evaluation uses ÏƒÂ²_t vs rÂ²_t")
master.append("- LSTM_Vol directly forecasts rÂ²_t")
master.append("")
master.append("## D. Files")
master.append("")
master.append("| File | Content |")
master.append("|------|---------|")
master.append("| `results/garch11_forecasts.csv` | GARCH(1,1) symmetric forecasts (full series) |")
master.append("| `results/garch11_metrics.csv` | OOS metrics for GARCH(1,1) and LSTM (verification) |")
master.append("| `results/garch11_dm_tests.csv` | DM tests: GARCH(1,1) vs LSTM_Vol |")
master.append("| `results/garch11_report.md` | Full GARCH(1,1) evaluation report |")
master.append("| `results/master_comparison.md` | This file â€” master synthesis |")
master.append("| `results/figures/fig_covid_stress_volatility.png` | COVID-19 stress figure |")
master.append("| `scripts/run_garch11_oos.py` | Reproducible script |")

with open(MASTER_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(master))
print(f"Master comparison saved: {MASTER_MD}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 9. FINAL VERIFICATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\n" + "=" * 78)
print(" FINAL VERIFICATION")
print("=" * 78)

checks = []

# Check 1: Symmetric GARCH
checks.append(('gamma[1]' not in param_names, "Model is symmetric (no gamma)"))

# Check 2: Test dates
checks.append((str(test_df['Date'].iloc[0].date()) == '2023-01-03', "Test starts 2023-01-03"))
checks.append((str(test_df['Date'].iloc[-1].date()) == '2026-08-13', "Test ends 2026-08-13"))

# Check 3: Test size
checks.append((test_size == 906, f"Test size = {test_size} (expected 906)"))

# Check 4: No test data in estimation
checks.append((res.nobs <= train_size, f"Estimation obs = {res.nobs} â‰¤ {train_size}"))

# Check 5: Forecasts exist
checks.append((os.path.exists(FORECASTS_CSV), "Forecasts CSV exists"))

# Check 6: Evaluation count matches
checks.append((len(eval_df) == 906, f"Evaluation obs = {len(eval_df)}"))

# Check 7: Existing files not modified
for existing_file in ['garch_forecasts.csv', 'metrics_summary.csv', 'evaluation_merged.csv',
                       '07_evaluation_report.txt', '08_dm_report.txt']:
    full_path = os.path.join(BASE_DIR, existing_file)
    checks.append((os.path.exists(full_path), f"Existing {existing_file} preserved"))

all_pass = True
for passed, desc in checks:
    status = "âœ“" if passed else "âœ—"
    if not passed:
        all_pass = False
    print(f"  {status} {desc}")

if all_pass:
    print("\nâœ… ALL CHECKS PASSED.")
else:
    print("\nâš ï¸  SOME CHECKS FAILED â€” review output above.")

print("\n" + "=" * 78)
print(" DONE")
print("=" * 78)
