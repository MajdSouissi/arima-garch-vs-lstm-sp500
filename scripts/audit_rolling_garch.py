import pandas as pd
import numpy as np
from scipy.stats import norm

# Load Data
file_path = r"c:\Users\LENOVO\Bureau\mastere\results_v2\rolling_garch\rolling_garch_predictions.csv"
df = pd.read_csv(file_path, parse_dates=['Date'])

print("=== 4. Verify rolling estimation itself ===")
# Just checking the dates we have
print(f"Total dates: {len(df)}")
print(f"First date: {df['Date'].min().date()}")
print(f"Last date: {df['Date'].max().date()}")

print("\n=== 5. Recompute the main metrics independently ===")
actual_var = df['Realized_Vol'].values
fixed_var = df['ARIMA_GARCH_Var'].values
stat_var = df['Rolling_GARCH_Var'].values
lstm_var = df['LSTM_Var'].values

def rmse(a, p): return np.sqrt(np.mean((a - p)**2))
def mae(a, p): return np.mean(np.abs(a - p))
def qlike(a, p):
    rv, fv = np.maximum(a, 1e-10), np.maximum(p, 1e-10)
    return np.mean(rv/fv - np.log(rv/fv) - 1)

print(f"Fixed GARCH   - RMSE: {rmse(actual_var, fixed_var):.6f}, MAE: {mae(actual_var, fixed_var):.6f}, QLIKE: {qlike(actual_var, fixed_var):.6f}")
print(f"Rolling GARCH - RMSE: {rmse(actual_var, stat_var):.6f}, MAE: {mae(actual_var, stat_var):.6f}, QLIKE: {qlike(actual_var, stat_var):.6f}")
print(f"LSTM_Vol      - RMSE: {rmse(actual_var, lstm_var):.6f}, MAE: {mae(actual_var, lstm_var):.6f}, QLIKE: {qlike(actual_var, lstm_var):.6f}")

def dm_test_vol(rv, fv1, fv2, loss='QLIKE'):
    rv, fv1, fv2 = np.maximum(rv, 1e-10), np.maximum(fv1, 1e-10), np.maximum(fv2, 1e-10)
    if loss == 'QLIKE':
        l1, l2 = rv/fv1 - np.log(rv/fv1) - 1, rv/fv2 - np.log(rv/fv2) - 1
    else:
        l1, l2 = (rv - fv1)**2, (rv - fv2)**2
    d = l1 - l2
    V = np.var(d, ddof=0) / len(d)
    if V <= 0: return 0.0, 1.0
    dm_stat = np.mean(d) / np.sqrt(V)
    p_val = 2.0 * (1.0 - norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_val

print("\nIndependent Full-Sample DM Tests (QLIKE):")
dm_f_r, p_f_r = dm_test_vol(actual_var, fixed_var, stat_var, 'QLIKE')
print(f"Fixed vs Rolling QLIKE: DM={dm_f_r:.6f}, p={p_f_r:.6e}")
dm_r_l, p_r_l = dm_test_vol(actual_var, stat_var, lstm_var, 'QLIKE')
print(f"Rolling vs LSTM QLIKE: DM={dm_r_l:.6f}, p={p_r_l:.6e}")

print("\n=== 1. Stress-window DM tests ===")
CRISIS_START = pd.to_datetime('2025-04-09')
CRISIS_END = pd.to_datetime('2025-05-08')
mask = (df['Date'] >= CRISIS_START) & (df['Date'] <= CRISIS_END)
df_stress = df[mask]
print(f"Stress window dates: {df_stress['Date'].min().date()} to {df_stress['Date'].max().date()} (n={len(df_stress)})")

s_actual_var = df_stress['Realized_Vol'].values
s_fixed_var = df_stress['ARIMA_GARCH_Var'].values
s_stat_var = df_stress['Rolling_GARCH_Var'].values
s_lstm_var = df_stress['LSTM_Var'].values

print(f"Stress Fixed GARCH QLIKE:   {qlike(s_actual_var, s_fixed_var):.6f}")
print(f"Stress Rolling GARCH QLIKE: {qlike(s_actual_var, s_stat_var):.6f}")
print(f"Stress LSTM_Vol QLIKE:      {qlike(s_actual_var, s_lstm_var):.6f}")

print("\nStress DM Tests (QLIKE):")
sdm_f_r, sp_f_r = dm_test_vol(s_actual_var, s_fixed_var, s_stat_var, 'QLIKE')
print(f"Fixed vs Rolling (QLIKE) - DM={sdm_f_r:.6f}, p={sp_f_r:.6e}")
sdm_r_l, sp_r_l = dm_test_vol(s_actual_var, s_stat_var, s_lstm_var, 'QLIKE')
print(f"Rolling vs LSTM (QLIKE) - DM={sdm_r_l:.6f}, p={sp_r_l:.6e}")

print("\nStress DM Tests (SE):")
sdm_f_r_se, sp_f_r_se = dm_test_vol(s_actual_var, s_fixed_var, s_stat_var, 'SE')
print(f"Fixed vs Rolling (SE) - DM={sdm_f_r_se:.6f}, p={sp_f_r_se:.6e}")
sdm_r_l_se, sp_r_l_se = dm_test_vol(s_actual_var, s_stat_var, s_lstm_var, 'SE')
print(f"Rolling vs LSTM (SE) - DM={sdm_r_l_se:.6f}, p={sp_r_l_se:.6e}")


print("\n=== 3. Audit alpha + beta at full precision ===")
persistence = df['persistence'].values
alpha = df['alpha'].values
beta = df['beta'].values

print(f"Median: {np.median(persistence):.15f}")
print(f"Min:    {np.min(persistence):.15f}")
print(f"Max:    {np.max(persistence):.15f}")

total = len(persistence)
c_999 = np.sum(persistence >= 0.999)
c_9999 = np.sum(persistence >= 0.9999)
c_1 = np.sum(persistence >= 1.0)
c_over_1 = np.sum(persistence > 1.0)

print(f">= 0.999:  {c_999} ({c_999/total*100:.2f}%)")
print(f">= 0.9999: {c_9999} ({c_9999/total*100:.2f}%)")
print(f">= 1.0000: {c_1} ({c_1/total*100:.2f}%)")
print(f">  1.0000: {c_over_1} ({c_over_1/total*100:.2f}%)")

if c_1 > 0:
    idx = np.where(persistence >= 1.0)[0][0]
    print(f"Sample where persistence >= 1.0:")
    print(f"  alpha: {alpha[idx]:.15f}")
    print(f"  beta:  {beta[idx]:.15f}")
    print(f"  sum:   {persistence[idx]:.15f}")

