"""
run_rolling_garch.py
Robustness Experiment: Rolling Estimation of ARIMA(2,0,0)-GARCH(1,1)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy.stats import norm
import shutil

# --- Paths ---
BASE_DIR = str(Path(__file__).resolve().parent.parent)
INPUT_CSV = os.path.join(BASE_DIR, "SP500_master_2000_2026.csv")
V2_EVAL_CSV = os.path.join(BASE_DIR, "results_v2", "evaluation_merged_v2.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results_v2", "rolling_garch")
MASTERS_RESULTS_DIR = os.path.join(BASE_DIR, "masters results only")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MASTERS_RESULTS_DIR, exist_ok=True)

W = 2000  # Rolling window size

def log(msg):
    print(msg)
    with open(os.path.join(RESULTS_DIR, "rolling_experiment_log.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear log
with open(os.path.join(RESULTS_DIR, "rolling_experiment_log.txt"), "w", encoding="utf-8") as f:
    f.write(f"Rolling GARCH Experiment started at {datetime.now()}\n")

log("Loading data...")
df_master = pd.read_csv(INPUT_CSV, parse_dates=['Date']).dropna(subset=['Log_Return']).reset_index(drop=True)
df_master = df_master.sort_values('Date').reset_index(drop=True)

df_v2 = pd.read_csv(V2_EVAL_CSV, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)

test_dates = df_v2['Date'].values
log(f"Number of test dates: {len(test_dates)}")

assert len(test_dates) == 906, f"Expected 906 test dates, found {len(test_dates)}"

rolling_results = []
failures = []

log(f"Starting rolling estimation with W = {W}...")

for i, t_date in enumerate(test_dates):
    # Slice master data up to t-1
    historical_data = df_master[df_master['Date'] < t_date]
    
    # Assert temporal ordering
    max_date = historical_data['Date'].max()
    assert max_date < t_date, f"Leakage detected! Max estimation date {max_date} is not before forecast date {t_date}"
    
    # Take last W observations
    window_data = historical_data.tail(W)
    window_returns = window_data['Log_Return'].values
    
    if len(window_returns) < W:
        log(f"Warning: Window at {t_date} has only {len(window_returns)} observations.")
        
    try:
        # Step 1: ARIMA(2,0,0)
        arima_model = ARIMA(window_returns, order=(2, 0, 0))
        res_arima = arima_model.fit()
        
        # 1-step ahead mean forecast
        mean_forecast = res_arima.forecast(steps=1)[0]
        
        # Residuals
        resid = res_arima.resid
        
        # Step 2: GARCH(1,1) Student-t
        garch_model = arch_model(resid, mean='Zero', vol='Garch', p=1, o=0, q=1, dist='t')
        res_garch = garch_model.fit(disp='off')
        
        # 1-step ahead variance forecast
        garch_forecast = res_garch.forecast(horizon=1)
        var_forecast = garch_forecast.variance.values[-1, 0]
        
        # Store params
        params = res_garch.params
        omega = params.get('omega', np.nan)
        alpha = params.get('alpha[1]', np.nan)
        beta = params.get('beta[1]', np.nan)
        nu = params.get('nu', np.nan)
        persistence = alpha + beta
        
        rolling_results.append({
            'Date': t_date,
            'Rolling_ARIMA_Mean': mean_forecast,
            'Rolling_GARCH_Var': var_forecast,
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'persistence': persistence,
            'nu': nu
        })
        
    except Exception as e:
        failures.append(t_date)
        log(f"Failed at {t_date}: {e}")
        # Forward-fill from previous successful step if available
        if len(rolling_results) > 0:
            last_res = rolling_results[-1]
            rolling_results.append({
                'Date': t_date,
                'Rolling_ARIMA_Mean': last_res['Rolling_ARIMA_Mean'],
                'Rolling_GARCH_Var': last_res['Rolling_GARCH_Var'],
                'omega': last_res['omega'],
                'alpha': last_res['alpha'],
                'beta': last_res['beta'],
                'persistence': last_res['persistence'],
                'nu': last_res['nu']
            })
        else:
            rolling_results.append({
                'Date': t_date,
                'Rolling_ARIMA_Mean': np.nan,
                'Rolling_GARCH_Var': np.nan,
                'omega': np.nan, 'alpha': np.nan, 'beta': np.nan, 'persistence': np.nan, 'nu': np.nan
            })

    if (i + 1) % 100 == 0:
        log(f"Completed {i + 1} / {len(test_dates)}")

log(f"Rolling estimation finished. Successful fits: {len(test_dates) - len(failures)}, Failures: {len(failures)}")
if len(failures) > 0:
    log(f"Failed dates: {failures}")

df_rolling = pd.DataFrame(rolling_results)

# Merge with existing evaluation data
df_eval = pd.merge(df_v2, df_rolling, on='Date', how='inner')
df_eval.to_csv(os.path.join(RESULTS_DIR, "rolling_garch_predictions.csv"), index=False)

# -- EVALUATION --
actual_ret = df_eval['Realized_Return'].values
actual_var = df_eval['Realized_Vol'].values

stat_mean = df_eval['Rolling_ARIMA_Mean'].values
stat_var  = df_eval['Rolling_GARCH_Var'].values
fixed_var = df_eval['ARIMA_GARCH_Var'].values
lstm_var  = df_eval['LSTM_Var'].values

def rmse(a, p): return np.sqrt(np.mean((a - p)**2))
def mae(a, p): return np.mean(np.abs(a - p))
def qlike(a, p):
    rv = np.maximum(a, 1e-10)
    fv = np.maximum(p, 1e-10)
    return np.mean(rv/fv - np.log(rv/fv) - 1)

# Metrics
metrics = []
# Fixed GARCH (Reference)
metrics.append({
    'Model': 'Fixed GARCH(1,1)',
    'RMSE': rmse(actual_var, fixed_var),
    'MAE': mae(actual_var, fixed_var),
    'QLIKE': qlike(actual_var, fixed_var)
})
# Rolling GARCH
metrics.append({
    'Model': 'Rolling GARCH(1,1)',
    'RMSE': rmse(actual_var, stat_var),
    'MAE': mae(actual_var, stat_var),
    'QLIKE': qlike(actual_var, stat_var)
})
# LSTM
metrics.append({
    'Model': 'LSTM_Vol',
    'RMSE': rmse(actual_var, lstm_var),
    'MAE': mae(actual_var, lstm_var),
    'QLIKE': qlike(actual_var, lstm_var)
})

df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv(os.path.join(RESULTS_DIR, "rolling_garch_metrics.csv"), index=False)
log("\n--- Metrics ---")
log(df_metrics.to_string())

# -- DIEBOLD-MARIANO TESTS --
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
    return dm_stat, 2.0 * (1.0 - norm.cdf(np.abs(dm_stat)))

dm_results = []
for loss in ['QLIKE', 'SE']:
    # 1. Fixed GARCH vs Rolling GARCH
    dm, p = dm_test_vol(actual_var, fixed_var, stat_var, loss)
    winner = "Fixed GARCH" if dm < 0 else "Rolling GARCH"
    conc = f"{winner} better" if p < 0.05 else "No sig diff"
    dm_results.append({'Comparison': 'Fixed GARCH vs Rolling GARCH', 'Loss': loss, 'DM_Stat': dm, 'p_value': p, 'Conclusion': conc})
    
    # 2. Rolling GARCH vs LSTM_Vol
    dm2, p2 = dm_test_vol(actual_var, stat_var, lstm_var, loss)
    winner2 = "Rolling GARCH" if dm2 < 0 else "LSTM_Vol"
    conc2 = f"{winner2} better" if p2 < 0.05 else "No sig diff"
    dm_results.append({'Comparison': 'Rolling GARCH vs LSTM_Vol', 'Loss': loss, 'DM_Stat': dm2, 'p_value': p2, 'Conclusion': conc2})

df_dm = pd.DataFrame(dm_results)
df_dm.to_csv(os.path.join(RESULTS_DIR, "rolling_garch_dm_tests.csv"), index=False)
log("\n--- DM Tests ---")
log(df_dm.to_string())

# -- PARAMETER SUMMARY & DIAGNOSTICS --
params_df = df_eval[['Date', 'omega', 'alpha', 'beta', 'persistence', 'nu']].copy()
param_stats = params_df[['omega', 'alpha', 'beta', 'persistence', 'nu']].describe()
param_stats.to_csv(os.path.join(RESULTS_DIR, "rolling_garch_parameter_summary.csv"))
log("\n--- Parameter Summary ---")
log(param_stats.to_string())

# Plot persistence
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(params_df['Date'], params_df['persistence'], color='purple', label='alpha + beta')
ax.axhline(1.0, color='red', linestyle='--', alpha=0.5)
ax.set_title('Rolling GARCH(1,1) Persistence over time')
ax.set_ylabel('alpha + beta')
ax.legend()
fig.savefig(os.path.join(RESULTS_DIR, "rolling_persistence_plot.png"), dpi=200, bbox_inches='tight')
plt.close()

# -- STRESS WINDOW --
CRISIS_START, CRISIS_END = pd.to_datetime('2025-04-09'), pd.to_datetime('2025-05-08')
stress_df = df_eval[(df_eval['Date'] >= CRISIS_START) & (df_eval['Date'] <= CRISIS_END)]
log(f"\n--- Stress Window ({CRISIS_START.date()} to {CRISIS_END.date()}) [n={len(stress_df)}] ---")

def qlike_series(a, p):
    rv, fv = np.maximum(a, 1e-10), np.maximum(p, 1e-10)
    return rv/fv - np.log(rv/fv) - 1

stress_df['QLIKE_Fixed'] = qlike_series(stress_df['Realized_Vol'].values, stress_df['ARIMA_GARCH_Var'].values)
stress_df['QLIKE_Rolling'] = qlike_series(stress_df['Realized_Vol'].values, stress_df['Rolling_GARCH_Var'].values)
stress_df['QLIKE_LSTM'] = qlike_series(stress_df['Realized_Vol'].values, stress_df['LSTM_Var'].values)

stress_metrics = pd.DataFrame({
    'Model': ['Fixed GARCH', 'Rolling GARCH', 'LSTM_Vol'],
    'Mean_QLIKE': [
        stress_df['QLIKE_Fixed'].mean(),
        stress_df['QLIKE_Rolling'].mean(),
        stress_df['QLIKE_LSTM'].mean()
    ]
})
stress_metrics.to_csv(os.path.join(RESULTS_DIR, "rolling_garch_stress_metrics.csv"), index=False)
log(stress_metrics.to_string())

log("Copying results to masters results only folder...")
for filename in os.listdir(RESULTS_DIR):
    src_file = os.path.join(RESULTS_DIR, filename)
    dst_file = os.path.join(MASTERS_RESULTS_DIR, filename)
    if os.path.isfile(src_file):
        shutil.copy2(src_file, dst_file)

log("Done.")
