"""
=============================================================================
run_experiment.py
Final Comprehensive Pipeline: ARIMA-GARCH vs LSTM
=============================================================================
This script executes the entire numerical pipeline as approved:
1. ARIMA(p,0,q) order selection on training data using BIC.
2. Two-step estimation of ARIMA(p,0,q) and symmetric GARCH(1,1)+t on train data.
3. Fixed-parameter out-of-sample (OOS) forecasting for the test set.
4. LSTM data preparation & training from scratch (Base Return & Volatility).
5. Forecast evaluation (RMSE, MAE, DirAcc, QLIKE) & DM tests.
6. Stress-testing analysis & publication-ready figure generation.

Note: No look-ahead bias. GARCH uses 1-step ahead conditional volatility
filtering. ARIMA uses 1-step ahead recursive apply.
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import itertools
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
import pickle

warnings.filterwarnings('ignore')

# ── Dependencies ─────────────────────────────────────────────────────────────
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = r"c:\Users\LENOVO\Bureau\mastere"
INPUT_CSV   = os.path.join(BASE_DIR, "SP500_master_2000_2026.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results_v2")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LSTM_DIR    = os.path.join(RESULTS_DIR, "lstm_data")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LSTM_DIR, exist_ok=True)

def log(msg):
    print(msg)
    with open(os.path.join(RESULTS_DIR, "experiment_log.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear log
with open(os.path.join(RESULTS_DIR, "experiment_log.txt"), "w", encoding="utf-8") as f:
    f.write(f"Experiment started at {datetime.now()}\n")

# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & SPLITTING
# ═════════════════════════════════════════════════════════════════════════════
log("=" * 80)
log("1. DATA LOADING & SPLITTING")
log("=" * 80)

df = pd.read_csv(INPUT_CSV, parse_dates=['Date'])
df = df.dropna(subset=['Log_Return']).reset_index(drop=True)

# Engineer enhanced features (for LSTM)
df['Return_Sq'] = df['Log_Return'] ** 2
df['Rolling_Vol_21'] = df['Log_Return'].rolling(window=21).std()
df['Rolling_MAE_21'] = df['Log_Return'].abs().rolling(window=21).mean()
# NOTE: Do NOT dropna() here — the rolling NaNs only affect LSTM features.
# ARIMA-GARCH does not use Rolling_Vol_21 / Rolling_MAE_21.
# The LSTM section handles its own NaN removal via df_lstm = df.dropna() (line ~219).

# Split dates
train_end_dt = pd.to_datetime('2018-12-31')
val_end_dt   = pd.to_datetime('2022-12-31')

train_mask = df['Date'] <= train_end_dt
val_mask   = (df['Date'] > train_end_dt) & (df['Date'] <= val_end_dt)
test_mask  = df['Date'] > val_end_dt

train_df = df[train_mask].copy()
val_df   = df[val_mask].copy()
test_df  = df[test_mask].copy()

log(f"Total Observations: {len(df)}")
log(f"Train: {train_df['Date'].iloc[0].date()} -> {train_df['Date'].iloc[-1].date()} ({len(train_df)} obs)")
log(f"Val:   {val_df['Date'].iloc[0].date()} -> {val_df['Date'].iloc[-1].date()} ({len(val_df)} obs)")
log(f"Test:  {test_df['Date'].iloc[0].date()} -> {test_df['Date'].iloc[-1].date()} ({len(test_df)} obs)")

returns = pd.Series(df['Log_Return'].values, index=df['Date'])
train_returns = returns.loc[:train_end_dt]


# ═════════════════════════════════════════════════════════════════════════════
# 2. ARIMA ORDER SELECTION (TRAINING SET ONLY)
# ═════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 80)
log("2. ARIMA ORDER SELECTION (TRAINING ONLY)")
log("=" * 80)

p_values = [0, 1, 2, 3]
q_values = [0, 1, 2, 3]

best_bic = np.inf
best_order = None
best_res_arima = None

log("Grid Search (p, 0, q) by BIC:")
for p, q in itertools.product(p_values, q_values):
    try:
        model = ARIMA(train_returns, order=(p, 0, q))
        res = model.fit()
        log(f"  ARIMA({p},0,{q}) - BIC: {res.bic:.2f} | AIC: {res.aic:.2f}")
        if res.bic < best_bic:
            best_bic = res.bic
            best_order = (p, 0, q)
            best_res_arima = res
    except Exception as e:
        log(f"  ARIMA({p},0,{q}) - FAILED")

log(f"\n> Selected Order by BIC: ARIMA{best_order} (BIC = {best_bic:.2f})")
log(best_res_arima.summary().as_text())

# Diagnostics on ARIMA residuals
arima_resid_train = best_res_arima.resid
lb_arima = acorr_ljungbox(arima_resid_train, lags=[10], return_df=True)
log("\nLjung-Box on ARIMA residuals:")
log(lb_arima.to_string())

arch_lm_arima = het_arch(arima_resid_train, nlags=10)
log(f"ARCH-LM on ARIMA residuals: p-value = {arch_lm_arima[1]:.4e}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. GARCH ESTIMATION (TWO-STEP, ON TRAIN RESIDUALS)
# ═════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 80)
log("3. GARCH ESTIMATION (TRAIN ONLY)")
log("=" * 80)
log("Fitting symmetric GARCH(1,1) + Student-t on ARIMA residuals...")

# We use mean='Zero' since the residuals are already mean-filtered by ARIMA
am = arch_model(arima_resid_train, mean='Zero', vol='Garch', p=1, o=0, q=1, dist='t')
res_garch = am.fit(disp='off')
log(res_garch.summary().as_text())

# Verify stationarity
alpha = res_garch.params.get('alpha[1]', 0)
beta  = res_garch.params.get('beta[1]', 0)
persistence = alpha + beta
log(f"\nGARCH Persistence (alpha + beta): {persistence:.6f}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. FIXED-PARAMETER OOS FORECASTS
# ═════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 80)
log("4. FIXED-PARAMETER OOS FORECASTS")
log("=" * 80)

# 4a. ARIMA Mean Forecasts
# res_arima.apply(returns, refit=False) calculates 1-step ahead forecasts (fittedvalues) 
# for the entire series using parameters fixed from the training fit.
full_arima_res = best_res_arima.apply(returns, refit=False)
pred_mean_full = full_arima_res.fittedvalues

# 4b. GARCH Volatility Forecasts
# We compute full series residuals from the ARIMA forecasts
full_residuals = returns - pred_mean_full

# am_full.fix() computes 1-step ahead conditional volatility (conditional_volatility)
# using the fixed GARCH parameters from the training fit.
am_full = arch_model(full_residuals, mean='Zero', vol='Garch', p=1, o=0, q=1, dist='t')
full_garch_res = am_full.fix(res_garch.params)
pred_var_full = full_garch_res.conditional_volatility ** 2

df['ARIMA_GARCH_Mean'] = pred_mean_full.values
df['ARIMA_GARCH_Var']  = pred_var_full.values

# Save full ARIMA-GARCH forecasts
df.to_csv(os.path.join(RESULTS_DIR, "arima_garch_full.csv"), index=False)
log("Generated full ARIMA-GARCH forecasts.")


# ═════════════════════════════════════════════════════════════════════════════
# 5. LSTM PREPARATION & TRAINING
# ═════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 80)
log("5. LSTM PREPARATION & TRAINING")
log("=" * 80)

# Targets for t+1
df['Target_Return'] = df['Log_Return'].shift(-1)
df['Target_Vol']    = df['Return_Sq'].shift(-1)  # Vol proxy: r²_{t+1}
df['Target_Date']   = df['Date'].shift(-1)

df_lstm = df.dropna().reset_index(drop=True)

feature_cols_base = ['Log_Return']
feature_cols_vol  = ['Log_Return', 'Return_Sq', 'Rolling_Vol_21', 'Rolling_MAE_21']

scaler_base = StandardScaler()
scaler_vol  = StandardScaler()

# Fit scalers on train only (where current date <= train_end_dt)
train_mask_lstm = df_lstm['Date'] <= train_end_dt
scaler_base.fit(df_lstm.loc[train_mask_lstm, feature_cols_base])
scaler_vol.fit(df_lstm.loc[train_mask_lstm, feature_cols_vol])

base_scaled = scaler_base.transform(df_lstm[feature_cols_base])
vol_scaled  = scaler_vol.transform(df_lstm[feature_cols_vol])

target_ret = df_lstm['Target_Return'].values
target_vol = df_lstm['Target_Vol'].values
target_dates = df_lstm['Target_Date'].values

LOOKBACK = 60
X_base, X_vol, y_ret, y_vol, seq_dates = [], [], [], [], []

for i in range(len(df_lstm) - LOOKBACK + 1):
    idx = i + LOOKBACK - 1
    X_base.append(base_scaled[i : idx + 1])
    X_vol.append(vol_scaled[i : idx + 1])
    y_ret.append(target_ret[idx])
    y_vol.append(target_vol[idx])
    seq_dates.append(target_dates[idx])

X_base = np.array(X_base)
X_vol  = np.array(X_vol)
y_ret  = np.array(y_ret)
y_vol  = np.array(y_vol)
seq_dates = pd.to_datetime(seq_dates)

train_idx = seq_dates <= train_end_dt
val_idx   = (seq_dates > train_end_dt) & (seq_dates <= val_end_dt)
test_idx  = seq_dates > val_end_dt

def get_split(mask):
    return X_base[mask], X_vol[mask], y_ret[mask], y_vol[mask], seq_dates[mask]

X_train_base, X_train_vol, y_train_ret, y_train_vol, _ = get_split(train_idx)
X_val_base,   X_val_vol,   y_val_ret,   y_val_vol,   _ = get_split(val_idx)
X_test_base,  X_test_vol,  y_test_ret,  y_test_vol,  test_dates = get_split(test_idx)

# LSTM Architecture
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=32, dropout=0.2, is_volatility=False):
        super().__init__()
        self.is_volatility = is_volatility
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        if self.is_volatility:
            out = self.softplus(out) + 1e-6
        return out.squeeze(-1)

def train_model(model, X_train, y_train, X_val, y_val, name="model"):
    model.to(DEVICE)
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    X_val_t, y_val_t = torch.FloatTensor(X_val).to(DEVICE), torch.FloatTensor(y_val).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 10
    
    for epoch in range(1, 101):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b.to(DEVICE)), y_b.to(DEVICE))
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            log(f"  {name} Early Stop at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
            break
            
    model.load_state_dict(best_state)
    return model

# Train LSTM_Return
log("Training LSTM_Return (Target: Return | Features: Base)...")
set_seeds()
lstm_ret_model = LSTMForecaster(input_size=X_train_base.shape[2], is_volatility=False)
lstm_ret_model = train_model(lstm_ret_model, X_train_base, y_train_ret, X_val_base, y_val_ret, "LSTM_Return")

# Train LSTM_Vol
log("Training LSTM_Vol (Target: Volatility | Features: Enhanced)...")
set_seeds()
lstm_vol_model = LSTMForecaster(input_size=X_train_vol.shape[2], is_volatility=True)
lstm_vol_model = train_model(lstm_vol_model, X_train_vol, y_train_vol, X_val_vol, y_val_vol, "LSTM_Vol")

# Generate Test Forecasts
lstm_ret_model.eval()
lstm_vol_model.eval()
with torch.no_grad():
    pred_lstm_ret = lstm_ret_model(torch.FloatTensor(X_test_base).to(DEVICE)).cpu().numpy()
    pred_lstm_vol = lstm_vol_model(torch.FloatTensor(X_test_vol).to(DEVICE)).cpu().numpy()

lstm_forecasts = pd.DataFrame({
    'Date': test_dates,
    'LSTM_Mean': pred_lstm_ret,
    'LSTM_Var': pred_lstm_vol
})


# ═════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION METRICS
# ═════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 80)
log("6. EVALUATION METRICS (TEST SET)")
log("=" * 80)

# Merge statistical and ML forecasts on common test dates
eval_df = pd.merge(
    df[['Date', 'Close', 'Log_Return', 'Return_Sq', 'ARIMA_GARCH_Mean', 'ARIMA_GARCH_Var']],
    lstm_forecasts,
    on='Date',
    how='inner'
)
eval_df.rename(columns={'Log_Return': 'Realized_Return', 'Return_Sq': 'Realized_Vol'}, inplace=True)
eval_df.to_csv(os.path.join(RESULTS_DIR, "evaluation_merged_v2.csv"), index=False)
log(f"Test period: {eval_df['Date'].iloc[0].date()} to {eval_df['Date'].iloc[-1].date()} ({len(eval_df)} obs)")

actual_ret = eval_df['Realized_Return'].values
actual_var = eval_df['Realized_Vol'].values

stat_mean = eval_df['ARIMA_GARCH_Mean'].values
stat_var  = eval_df['ARIMA_GARCH_Var'].values
lstm_mean = eval_df['LSTM_Mean'].values
lstm_var  = eval_df['LSTM_Var'].values

def rmse(a, p): return np.sqrt(np.mean((a - p)**2))
def mae(a, p): return np.mean(np.abs(a - p))
def diracc(a, p): return np.mean(np.sign(a) == np.sign(p)) * 100
def qlike(a, p):
    rv = np.maximum(a, 1e-10)
    fv = np.maximum(p, 1e-10)
    return np.mean(rv/fv - np.log(rv/fv) - 1)

metrics = []

# Returns
r_rmse_stat, r_mae_stat, r_da_stat = rmse(actual_ret, stat_mean), mae(actual_ret, stat_mean), diracc(actual_ret, stat_mean)
r_rmse_lstm, r_mae_lstm, r_da_lstm = rmse(actual_ret, lstm_mean), mae(actual_ret, lstm_mean), diracc(actual_ret, lstm_mean)
metrics.append({'Task': 'Return', 'Model': 'ARIMA-GARCH', 'RMSE': r_rmse_stat, 'MAE': r_mae_stat, 'DirAcc': r_da_stat, 'QLIKE': np.nan})
metrics.append({'Task': 'Return', 'Model': 'LSTM_Return', 'RMSE': r_rmse_lstm, 'MAE': r_mae_lstm, 'DirAcc': r_da_lstm, 'QLIKE': np.nan})

log(f"Returns  - ARIMA-GARCH : RMSE={r_rmse_stat:.4f}, MAE={r_mae_stat:.4f}, DirAcc={r_da_stat:.2f}%")
log(f"Returns  - LSTM_Return : RMSE={r_rmse_lstm:.4f}, MAE={r_mae_lstm:.4f}, DirAcc={r_da_lstm:.2f}%")

# Volatility
v_rmse_stat, v_mae_stat, v_qlike_stat = rmse(actual_var, stat_var), mae(actual_var, stat_var), qlike(actual_var, stat_var)
v_rmse_lstm, v_mae_lstm, v_qlike_lstm = rmse(actual_var, lstm_var), mae(actual_var, lstm_var), qlike(actual_var, lstm_var)
metrics.append({'Task': 'Volatility', 'Model': 'GARCH(1,1)', 'RMSE': v_rmse_stat, 'MAE': v_mae_stat, 'DirAcc': np.nan, 'QLIKE': v_qlike_stat})
metrics.append({'Task': 'Volatility', 'Model': 'LSTM_Vol', 'RMSE': v_rmse_lstm, 'MAE': v_mae_lstm, 'DirAcc': np.nan, 'QLIKE': v_qlike_lstm})

log(f"Vol      - GARCH(1,1)  : RMSE={v_rmse_stat:.4f}, MAE={v_mae_stat:.4f}, QLIKE={v_qlike_stat:.4f}")
log(f"Vol      - LSTM_Vol    : RMSE={v_rmse_lstm:.4f}, MAE={v_mae_lstm:.4f}, QLIKE={v_qlike_lstm:.4f}")

pd.DataFrame(metrics).to_csv(os.path.join(RESULTS_DIR, "metrics_v2.csv"), index=False)


# ═════════════════════════════════════════════════════════════════════════════
# 7. DIEBOLD-MARIANO TESTS
# ═════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 80)
log("7. DIEBOLD-MARIANO TESTS")
log("=" * 80)

def dm_test(e1, e2, loss='SE'):
    d = (e1**2 - e2**2) if loss == 'SE' else (np.abs(e1) - np.abs(e2))
    V = np.var(d, ddof=0) / len(d)
    if V <= 0: return 0.0, 1.0
    dm_stat = np.mean(d) / np.sqrt(V)
    return dm_stat, 2.0 * (1.0 - norm.cdf(np.abs(dm_stat)))

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

e_stat = actual_ret - stat_mean
e_lstm = actual_ret - lstm_mean

dm_results = []
for loss in ['SE', 'AE']:
    dm, p = dm_test(e_stat, e_lstm, loss)
    winner = "ARIMA-GARCH" if dm < 0 else "LSTM_Return"
    conc = f"{winner} better" if p < 0.05 else "No sig diff"
    dm_results.append({'Task': 'Return', 'Loss': loss, 'DM_Stat': dm, 'p_value': p, 'Conclusion': conc})
    log(f"Return DM ({loss}): DM={dm:+.4f}, p={p:.4f} -> {conc}")

for loss in ['QLIKE', 'SE']:
    dm, p = dm_test_vol(actual_var, stat_var, lstm_var, loss)
    winner = "GARCH(1,1)" if dm < 0 else "LSTM_Vol"
    conc = f"{winner} better" if p < 0.05 else "No sig diff"
    dm_results.append({'Task': 'Volatility', 'Loss': loss, 'DM_Stat': dm, 'p_value': p, 'Conclusion': conc})
    log(f"Vol DM ({loss}): DM={dm:+.4f}, p={p:.4f} -> {conc}")

pd.DataFrame(dm_results).to_csv(os.path.join(RESULTS_DIR, "dm_tests_v2.csv"), index=False)


# ═════════════════════════════════════════════════════════════════════════════
# 8. STRESS / CRISIS ANALYSIS & PLOTS
# ═════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 80)
log("8. STRESS ANALYSIS & FIGURES")
log("=" * 80)

CRISIS_START, CRISIS_END = pd.to_datetime('2025-04-09'), pd.to_datetime('2025-05-08')
def shade_crisis(ax): ax.axvspan(CRISIS_START, CRISIS_END, color='red', alpha=0.1, label='Crisis Window')

dates = eval_df['Date']
def qlike_series(a, p):
    rv, fv = np.maximum(a, 1e-10), np.maximum(p, 1e-10)
    return rv/fv - np.log(rv/fv) - 1

eval_df['QLIKE_Stat'] = qlike_series(actual_var, stat_var)
eval_df['QLIKE_LSTM'] = qlike_series(actual_var, lstm_var)

# Fig 1: Returns
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates, actual_ret, color='gray', linewidth=0.5, label='Actual')
ax.plot(dates, stat_mean, color='blue', linewidth=1.0, alpha=0.8, label='ARIMA-GARCH')
ax.plot(dates, lstm_mean, color='orange', linewidth=1.0, alpha=0.8, label='LSTM_Return')
shade_crisis(ax)
ax.set_title('Out-of-Sample Return Forecasts')
ax.legend()
fig.savefig(os.path.join(FIGURES_DIR, 'fig1_returns.png'), dpi=200, bbox_inches='tight')
plt.close()

# Fig 2: Volatility
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates, np.sqrt(actual_var), color='gray', linewidth=0.5, alpha=0.5, label='|r| (proxy)')
ax.plot(dates, np.sqrt(stat_var), color='blue', linewidth=1.0, label='GARCH(1,1) σ')
ax.plot(dates, np.sqrt(lstm_var), color='orange', linewidth=1.0, label='LSTM_Vol σ')
shade_crisis(ax)
ax.set_title('Out-of-Sample Volatility Forecasts')
ax.legend()
fig.savefig(os.path.join(FIGURES_DIR, 'fig2_volatility.png'), dpi=200, bbox_inches='tight')
plt.close()

# Fig 3: Rolling QLIKE
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates, eval_df['QLIKE_Stat'].rolling(21).mean(), color='blue', label='GARCH(1,1)')
ax.plot(dates, eval_df['QLIKE_LSTM'].rolling(21).mean(), color='orange', label='LSTM_Vol')
shade_crisis(ax)
ax.set_title('21-Day Rolling QLIKE Loss')
ax.legend()
fig.savefig(os.path.join(FIGURES_DIR, 'fig3_rolling_qlike.png'), dpi=200, bbox_inches='tight')
plt.close()

# Crisis Summary
mask_crisis = (dates >= CRISIS_START) & (dates <= CRISIS_END)
log(f"\n--- Crisis Window ({CRISIS_START.date()} to {CRISIS_END.date()}) ---")
log(f"GARCH Mean QLIKE: {eval_df.loc[mask_crisis, 'QLIKE_Stat'].mean():.4f}")
log(f"LSTM  Mean QLIKE: {eval_df.loc[mask_crisis, 'QLIKE_LSTM'].mean():.4f}")

# Markdown report
md = [
    "# ARIMA-GARCH vs LSTM - Final Experiment Results",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "\n## 1. ARIMA Order Selection",
    f"Selected Order (BIC): ARIMA{best_order}",
    "This was fit on the training data only.",
    "\n## 2. Evaluation Metrics",
    pd.DataFrame(metrics).to_markdown(index=False),
    "\n## 3. Diebold-Mariano Tests",
    pd.DataFrame(dm_results).to_markdown(index=False),
    "\n## 4. Methodological Notes",
    "- 100% fixed parameters out-of-sample (no expanding window re-estimation).",
    "- Two-step estimation used for ARIMA-GARCH.",
    "- LSTMs retrained from scratch.",
    "- No look-ahead bias, fully causal filters used."
]
with open(os.path.join(RESULTS_DIR, "experiment_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))

log("\nExperiment Complete. Check results_v2 directory.")
