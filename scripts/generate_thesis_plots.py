import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import matplotlib.dates as mdates

# Set matplotlib styling
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 200

def compute_qlike(realized_var, forecast_var):
    eps = 1e-10
    rv = np.maximum(realized_var, eps)
    fv = np.maximum(forecast_var, eps)
    ratio = rv / fv
    return ratio - np.log(ratio) - 1

# Load Data
garch_df = pd.read_csv('results/garch11_forecasts.csv', parse_dates=['Date'])
garch_df = garch_df[garch_df['Split'] == 'Test'].copy()

lstm_df = pd.read_csv('lstm_forecasts.csv', parse_dates=['Date'])

# Merge
df = pd.merge(garch_df, lstm_df, on='Date', how='inner').sort_values('Date').reset_index(drop=True)

# Compute variables
df['GARCH11_VarForecast'] = df['GARCH11_Vol'] ** 2
df['QLIKE_GARCH'] = compute_qlike(df['Realized_Vol_Proxy'], df['GARCH11_VarForecast'])
df['QLIKE_LSTM'] = compute_qlike(df['Realized_Vol_Proxy'], df['LSTM_M3_Var'])

# Definitions
CRISIS_START = pd.to_datetime('2025-04-09')
CRISIS_END = pd.to_datetime('2025-05-08')

out_dir = 'report_ready_plots'
os.makedirs(out_dir, exist_ok=True)

def shade_crisis(ax):
    ax.axvspan(CRISIS_START, CRISIS_END, color='red', alpha=0.1, label='FenÃªtre de crise')

# ---------------------------------------------------------
# Figure 1: Returns
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['Date'], df['Log_Return'], color='gray', linewidth=0.5, label='Rendement rÃ©alisÃ©')
ax.plot(df['Date'], df['GARCH11_Mean'], color='blue', linewidth=1.5, label='GARCH(1,1) Î¼')
ax.plot(df['Date'], df['LSTM_M1_Mean'], color='orange', linewidth=1.5, label='LSTM_Base')
shade_crisis(ax)
ax.set_ylabel('Rendement (%)')
ax.legend()
plt.tight_layout()
fig.savefig(f'{out_dir}/fig_returns_test.png')
plt.close(fig)

# ---------------------------------------------------------
# Figure 2: Forecast Errors
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

err_garch_ret = df['Log_Return'] - df['GARCH11_Mean']
err_lstm_ret = df['Log_Return'] - df['LSTM_M1_Mean']
ax1.plot(df['Date'], err_garch_ret, color='blue', alpha=0.7, label='GARCH(1,1)', linewidth=1)
ax1.plot(df['Date'], err_lstm_ret, color='orange', alpha=0.7, label='LSTM_Base', linewidth=1)
ax1.set_title('Erreur de prÃ©vision â€” rendements')
shade_crisis(ax1)
ax1.legend()

err_garch_vol = df['Realized_Vol_Proxy'] - df['GARCH11_VarForecast']
err_lstm_vol = df['Realized_Vol_Proxy'] - df['LSTM_M3_Var']
ax2.plot(df['Date'], err_garch_vol, color='blue', alpha=0.7, label='GARCH(1,1)', linewidth=1)
ax2.plot(df['Date'], err_lstm_vol, color='orange', alpha=0.7, label='LSTM_Vol', linewidth=1)
ax2.set_title('Erreur de prÃ©vision â€” volatilitÃ©')
shade_crisis(ax2)
ax2.legend()

plt.tight_layout()
fig.savefig(f'{out_dir}/fig_errors_test.png')
plt.close(fig)

# ---------------------------------------------------------
# Figure 3: Rolling QLIKE
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
qlike_garch_rolling = df['QLIKE_GARCH'].rolling(21).mean()
qlike_lstm_rolling = df['QLIKE_LSTM'].rolling(21).mean()
ax.plot(df['Date'], qlike_garch_rolling, color='blue', label='GARCH(1,1)')
ax.plot(df['Date'], qlike_lstm_rolling, color='orange', label='LSTM_Vol')
shade_crisis(ax)
ax.set_ylabel('QLIKE (moyenne mobile 21 jours)')
ax.legend()
plt.tight_layout()
fig.savefig(f'{out_dir}/fig_qlike_rolling.png')
plt.close(fig)

# ---------------------------------------------------------
# Figure 4: QLIKE Comparison
# ---------------------------------------------------------
crisis_mask = (df['Date'] >= CRISIS_START) & (df['Date'] <= CRISIS_END)
not_crisis_mask = ~crisis_mask

mean_garch = [
    df['QLIKE_GARCH'].mean(),
    df.loc[crisis_mask, 'QLIKE_GARCH'].mean(),
    df.loc[not_crisis_mask, 'QLIKE_GARCH'].mean()
]
mean_lstm = [
    df['QLIKE_LSTM'].mean(),
    df.loc[crisis_mask, 'QLIKE_LSTM'].mean(),
    df.loc[not_crisis_mask, 'QLIKE_LSTM'].mean()
]

labels = ['PÃ©riode complÃ¨te', 'FenÃªtre de crise', 'Hors crise']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, mean_garch, width, label='GARCH(1,1)', color='blue')
ax.bar(x + width/2, mean_lstm, width, label='LSTM_Vol', color='orange')
ax.set_ylabel('QLIKE moyen')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
fig.savefig(f'{out_dir}/fig_qlike_windows_bar.png')
plt.close(fig)

# ---------------------------------------------------------
# Figure 5: Calibration Scatter
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

max_val = max(df['Realized_Vol_Proxy'].max(), df['GARCH11_VarForecast'].max(), df['LSTM_M3_Var'].max())

ax1.scatter(df['Realized_Vol_Proxy'], df['GARCH11_VarForecast'], alpha=0.5, color='blue')
ax1.plot([0, max_val], [0, max_val], 'k--', color='gray')
ax1.set_title('GARCH(1,1)')
ax1.set_xlabel('Realized_Vol_Proxy')
ax1.set_ylabel('Variance prÃ©vue')

ax2.scatter(df['Realized_Vol_Proxy'], df['LSTM_M3_Var'], alpha=0.5, color='orange')
ax2.plot([0, max_val], [0, max_val], 'k--', color='gray')
ax2.set_title('LSTM_Vol')
ax2.set_xlabel('Realized_Vol_Proxy')
ax2.set_ylabel('Variance prÃ©vue')

plt.tight_layout()
fig.savefig(f'{out_dir}/fig_calibration_scatter.png')
plt.close(fig)

# ---------------------------------------------------------
# Figure 6: Histogram of Errors
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(err_garch_ret, bins=60, alpha=0.5, label='GARCH(1,1)', color='blue')
ax.hist(err_lstm_ret, bins=60, alpha=0.5, label='LSTM_Base', color='orange')
ax.axvline(0, color='black', linestyle='--')
ax.legend()
plt.tight_layout()
fig.savefig(f'{out_dir}/fig_error_distribution_returns.png')
plt.close(fig)

# ---------------------------------------------------------
# Figure 7: DM Statistics Bar Chart
# ---------------------------------------------------------
dm_labels = ['Rendements (SE)', 'Rendements (AE)', 'VolatilitÃ© (SE)', 'VolatilitÃ© (QLIKE)']
dm_stats = [0.1511, -2.1787, 0.5879, 3.6770]
p_values = [0.8799, 0.0294, 0.5566, 0.0002]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['gray' if p > 0.05 else 'darkblue' for p in p_values]
bars = ax.bar(dm_labels, dm_stats, color=colors)

ax.set_ylabel('Statistique DM')
ax.axhline(0, color='black', linewidth=1)

for bar, p in zip(bars, p_values):
    height = bar.get_height()
    y_pos = height + 0.1 if height > 0 else height - 0.2
    ax.annotate(f'p={p}',
                xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                ha='center', va='bottom' if height > 0 else 'top')

plt.tight_layout()
fig.savefig(f'{out_dir}/fig_dm_summary_bar.png')
plt.close(fig)

# ---------------------------------------------------------
# Figure 8: Crisis Zoom
# ---------------------------------------------------------
zoom_start = pd.to_datetime('2025-03-15')
zoom_end = pd.to_datetime('2025-06-01')
zoom_df = df[(df['Date'] >= zoom_start) & (df['Date'] <= zoom_end)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(zoom_df['Date'], zoom_df['Log_Return'], color='gray', linewidth=1, marker='.', label='RÃ©alisÃ©')
ax1.plot(zoom_df['Date'], zoom_df['GARCH11_Mean'], color='blue', linewidth=1.5, marker='.', label='GARCH(1,1)')
ax1.plot(zoom_df['Date'], zoom_df['LSTM_M1_Mean'], color='orange', linewidth=1.5, marker='.', label='LSTM_Base')
ax1.axvspan(CRISIS_START, CRISIS_END, color='red', alpha=0.1, label='FenÃªtre de crise')
ax1.set_title('Rendements')
ax1.legend()

ax2.plot(zoom_df['Date'], np.sqrt(zoom_df['Realized_Vol_Proxy']), color='gray', linewidth=1, marker='.', label='RÃ©alisÃ©')
ax2.plot(zoom_df['Date'], np.sqrt(zoom_df['GARCH11_VarForecast']), color='blue', linewidth=1.5, marker='.', label='GARCH(1,1)')
ax2.plot(zoom_df['Date'], np.sqrt(zoom_df['LSTM_M3_Var']), color='orange', linewidth=1.5, marker='.', label='LSTM_Vol')
ax2.axvspan(CRISIS_START, CRISIS_END, color='red', alpha=0.1, label='FenÃªtre de crise')
ax2.set_title('VolatilitÃ© (Ïƒ)')
ax2.legend()

# Format x-axis for dates properly
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig(f'{out_dir}/fig_crisis_zoom.png')
plt.close(fig)

print("All figures generated successfully.")
