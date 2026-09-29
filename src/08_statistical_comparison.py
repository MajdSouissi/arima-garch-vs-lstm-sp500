"""
=============================================================================
08_statistical_comparison.py
Diebold-Mariano Tests
=============================================================================
H0: Both forecasts have equal predictive accuracy.
H1: The two forecasts have different expected loss.

DM stat < 0 â†’ Model 1 has lower loss (better).
DM stat > 0 â†’ Model 2 has lower loss (better).
=============================================================================
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.stats import norm

BASE_DIR = str(Path(__file__).resolve().parent.parent)
EVAL_CSV   = os.path.join(BASE_DIR, "evaluation_merged.csv")
DM_CSV     = os.path.join(BASE_DIR, "dm_tests.csv")
REPORT_TXT = os.path.join(BASE_DIR, "08_dm_report.txt")

# â”€â”€ Diebold-Mariano Test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def dm_test(e1, e2, loss='SE', h=1):
    """
    Diebold-Mariano test.
    e1, e2: forecast errors from model 1 and model 2.
    loss: 'SE' (squared error) or 'AE' (absolute error).
    h: forecast horizon (1 for 1-step-ahead).
    
    Returns: (DM statistic, two-sided p-value)
    """
    if loss == 'SE':
        d = e1**2 - e2**2
    elif loss == 'AE':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    n = len(d)
    d_bar = np.mean(d)

    # Autocovariance (Newey-West truncation at h-1 lags)
    gamma_0 = np.var(d, ddof=0)
    if gamma_0 < 1e-15:
        return 0.0, 1.0

    # For h=1, standard DM/Newey-West truncation uses h-1=0 because there are no 
    # overlapping forecast horizons. (Note: errors can still exhibit serial 
    # dependence under model misspecification, but lag 0 is standard for h=1).
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


# â”€â”€ Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Loading evaluation data...")
df = pd.read_csv(EVAL_CSV, parse_dates=['Date'])
print(f"Evaluation period: {df['Date'].min().date()} â†’ {df['Date'].max().date()}")
print(f"Observations: {len(df)}")

actual_ret = df['Realized_Return'].values
actual_vol = df['Realized_Vol_Proxy'].values

# Forecast errors (actual - predicted)
e_garch_const = actual_ret - df['GARCH_M1_Mean'].values
e_garch_ar1   = actual_ret - df['GARCH_M2_Mean'].values
e_lstm_base   = actual_ret - df['LSTM_M1_Mean'].values
e_lstm_enh    = actual_ret - df['LSTM_M2_Mean'].values

# Variance forecasts
var_garch_const = df['GARCH_M1_Vol'].values**2
var_garch_ar1   = df['GARCH_M2_Vol'].values**2
var_lstm        = df['LSTM_M3_Var'].values  # LSTM M3 already outputs variance proxy r^2

results = []

def run_dm(task, m1_name, e1, m2_name, e2, loss):
    dm, p = dm_test(e1, e2, loss=loss)
    if p < 0.05:
        winner = m1_name if dm < 0 else m2_name
        conclusion = f"{winner} significantly better (p={p:.4f})"
    else:
        conclusion = f"No significant difference (p={p:.4f})"
    results.append({
        'Task': task, 'Model_1': m1_name, 'Model_2': m2_name,
        'Loss': loss, 'DM_Stat': round(dm, 4), 'p_value': round(p, 4),
        'Conclusion': conclusion
    })
    print(f"  {m1_name} vs {m2_name} ({loss}): DM={dm:+.4f}  p={p:.4f}  â†’ {conclusion}")

def run_dm_vol(m1_name, v1, m2_name, v2, loss):
    dm, p = dm_test_vol(actual_vol, v1, v2, loss=loss)
    if p < 0.05:
        winner = m1_name if dm < 0 else m2_name
        conclusion = f"{winner} significantly better (p={p:.4f})"
    else:
        conclusion = f"No significant difference (p={p:.4f})"
    results.append({
        'Task': 'Volatility', 'Model_1': m1_name, 'Model_2': m2_name,
        'Loss': loss, 'DM_Stat': round(dm, 4), 'p_value': round(p, 4),
        'Conclusion': conclusion
    })
    print(f"  {m1_name} vs {m2_name} ({loss}): DM={dm:+.4f}  p={p:.4f}  â†’ {conclusion}")

# â”€â”€ Return Comparisons â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n--- Diebold-Mariano: RETURN FORECASTING ---")

# GARCH Const vs LSTM Base
run_dm('Return', 'GARCH_Const', e_garch_const, 'LSTM_Base', e_lstm_base, 'SE')
run_dm('Return', 'GARCH_Const', e_garch_const, 'LSTM_Base', e_lstm_base, 'AE')

# GARCH AR(1) vs LSTM Enhanced
run_dm('Return', 'GARCH_AR1', e_garch_ar1, 'LSTM_Enh', e_lstm_enh, 'SE')
run_dm('Return', 'GARCH_AR1', e_garch_ar1, 'LSTM_Enh', e_lstm_enh, 'AE')

# GARCH Const vs GARCH AR(1)  (sanity check)
run_dm('Return', 'GARCH_Const', e_garch_const, 'GARCH_AR1', e_garch_ar1, 'SE')

# LSTM Base vs LSTM Enhanced
run_dm('Return', 'LSTM_Base', e_lstm_base, 'LSTM_Enh', e_lstm_enh, 'SE')

# â”€â”€ Volatility Comparisons â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n--- Diebold-Mariano: VOLATILITY FORECASTING ---")

run_dm_vol('GARCH_Const', var_garch_const, 'LSTM_Vol', var_lstm, 'QLIKE')
run_dm_vol('GARCH_Const', var_garch_const, 'LSTM_Vol', var_lstm, 'SE')
run_dm_vol('GARCH_AR1',   var_garch_ar1,   'LSTM_Vol', var_lstm, 'QLIKE')
run_dm_vol('GARCH_AR1',   var_garch_ar1,   'LSTM_Vol', var_lstm, 'SE')

# â”€â”€ Save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
dm_df = pd.DataFrame(results)
dm_df.to_csv(DM_CSV, index=False)
print(f"\nDM test results saved to: {DM_CSV}")

with open(REPORT_TXT, 'w', encoding='utf-8') as f:
    f.write("DIEBOLD-MARIANO TEST RESULTS\n")
    f.write("="*80 + "\n")
    f.write("H0: Equal predictive accuracy.\n")
    f.write("H1: Different expected forecast loss.\n")
    f.write("DM < 0 â†’ Model 1 better.  DM > 0 â†’ Model 2 better.\n")
    f.write("Significance level: Î± = 0.05 (two-sided).\n\n")
    f.write(dm_df.to_string(index=False))
print(f"Report saved to: {REPORT_TXT}")
print("\nâœ… Diebold-Mariano tests complete.")
