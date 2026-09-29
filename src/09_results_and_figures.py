"""
=============================================================================
09_results_and_figures.py
Final Publication-Quality Tables and Figures
=============================================================================
Generates Tables 1â€“6 and Figures 1â€“6 for the Master's thesis.
=============================================================================
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = str(Path(__file__).resolve().parent.parent)
EVAL_CSV    = os.path.join(BASE_DIR, "evaluation_merged.csv")
GARCH_CSV   = os.path.join(BASE_DIR, "garch_forecasts.csv")
METRICS_CSV = os.path.join(BASE_DIR, "metrics_summary.csv")
DM_CSV      = os.path.join(BASE_DIR, "dm_tests.csv")
HYPER_CSV   = os.path.join(BASE_DIR, "lstm_hyperparameters.csv")
REPORT_MD   = os.path.join(BASE_DIR, "09_final_research_summary.md")
PLOTS_DIR   = os.path.join(BASE_DIR, "final_results_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 11, 'axes.facecolor': 'white', 'figure.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3,
})

# â”€â”€ Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
eval_df    = pd.read_csv(EVAL_CSV, parse_dates=['Date'])
garch_df   = pd.read_csv(GARCH_CSV, parse_dates=['Date'])
metrics_df = pd.read_csv(METRICS_CSV)
dm_df      = pd.read_csv(DM_CSV)

dates = eval_df['Date']
actual_ret = eval_df['Realized_Return']

# Previous close for price reconstruction
prev_close = eval_df['Close'].shift(1).copy()
first_eval_date = eval_df['Date'].iloc[0]
pre_row = garch_df[garch_df['Date'] < first_eval_date].iloc[-1]
prev_close.iloc[0] = pre_row['Close']
actual_price = eval_df['Close']

# â”€â”€ TABLE 1: Statistical Model Comparison (from GARCH report) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Parse from 04_garch_report.txt
garch_report = os.path.join(BASE_DIR, "04_garch_report.txt")
table1_data = []
if os.path.exists(garch_report):
    with open(garch_report, 'r') as f:
        text = f.read()
    # Extract AIC/BIC/LL for each model
    import re
    for model_name, label in [('Constant Mean', 'Const+GJR-GARCH(1,1)+t'),
                               ('AR - GJR', 'AR(1)+GJR-GARCH(1,1)+t')]:
        aic_m = re.search(r'AIC:\s+([\d.]+)', text[text.find(model_name):])
        bic_m = re.search(r'BIC:\s+([\d.]+)', text[text.find(model_name):])
        ll_m  = re.search(r'Log-Likelihood:\s+([-\d.]+)', text[text.find(model_name):])
        if aic_m and bic_m and ll_m:
            table1_data.append({
                'Model': label,
                'AIC': float(aic_m.group(1)),
                'BIC': float(bic_m.group(1)),
                'Log-Likelihood': float(ll_m.group(1))
            })

# â”€â”€ FIGURES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Figure 1: Actual vs Predicted Returns
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(dates, actual_ret, color='gray', alpha=0.4, linewidth=0.4, label='Actual')
ax.plot(dates, eval_df['GARCH_M2_Mean'], color='#1565C0', linewidth=0.8, alpha=0.8, label='GARCH AR(1)')
ax.plot(dates, eval_df['LSTM_M2_Mean'], color='#E65100', linewidth=0.8, alpha=0.8, label='LSTM Enhanced')
ax.axhline(0, color='black', linewidth=0.3)
ax.set_title('Figure 1: Out-of-Sample Return Forecasts')
ax.set_ylabel('Log Return (%)')
ax.set_xlabel('Date')
ax.legend(loc='upper right')
fig.savefig(os.path.join(PLOTS_DIR, 'fig1_actual_vs_pred_returns.png'))
plt.close()

# Figure 2: Actual vs Predicted Prices
garch_price = prev_close * np.exp(eval_df['GARCH_M2_Mean'] / 100.0)
lstm_price  = prev_close * np.exp(eval_df['LSTM_M2_Mean'] / 100.0)

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(dates, actual_price, color='black', linewidth=1.2, label='Actual Price')
ax.plot(dates, garch_price, color='#1565C0', linewidth=0.8, linestyle='--', alpha=0.8, label='GARCH Price')
ax.plot(dates, lstm_price, color='#E65100', linewidth=0.8, linestyle='--', alpha=0.8, label='LSTM Price')
ax.set_title('Figure 2: Out-of-Sample Price Forecasts')
ax.set_ylabel('S&P 500 Index')
ax.set_xlabel('Date')
ax.legend(loc='upper left')
fig.savefig(os.path.join(PLOTS_DIR, 'fig2_actual_vs_pred_prices.png'))
plt.close()

# Figure 3: Actual vs Predicted Volatility
realized_vol = np.sqrt(np.maximum(eval_df['Realized_Vol_Proxy'], 0))
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(dates, realized_vol, color='gray', alpha=0.35, linewidth=0.4, label='|r| (proxy)')
ax.plot(dates, eval_df['GARCH_M2_Vol'], color='#1565C0', linewidth=1.0, label='GARCH Ïƒ_t')
ax.plot(dates, np.sqrt(eval_df['LSTM_M3_Var']), color='#E65100', linewidth=1.0, label='LSTM Vol')
ax.set_title('Figure 3: Out-of-Sample Volatility Forecasts')
ax.set_ylabel('Volatility')
ax.set_xlabel('Date')
ax.legend(loc='upper right')
fig.savefig(os.path.join(PLOTS_DIR, 'fig3_actual_vs_pred_volatility.png'))
plt.close()

# Figure 4: Forecast Errors Over Time
garch_err = actual_ret - eval_df['GARCH_M2_Mean']
lstm_err  = actual_ret - eval_df['LSTM_M2_Mean']

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
axes[0].plot(dates, garch_err, color='#1565C0', alpha=0.6, linewidth=0.4)
axes[0].axhline(0, color='black', linewidth=0.3)
axes[0].set_title('Figure 4a: GARCH AR(1) Forecast Errors')
axes[0].set_ylabel('Error (%)')

axes[1].plot(dates, lstm_err, color='#E65100', alpha=0.6, linewidth=0.4)
axes[1].axhline(0, color='black', linewidth=0.3)
axes[1].set_title('Figure 4b: LSTM Enhanced Forecast Errors')
axes[1].set_ylabel('Error (%)')
axes[1].set_xlabel('Date')
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, 'fig4_forecast_errors.png'))
plt.close()

# Figure 5: LSTM Training Curves (copy from lstm_plots if exists)
lstm_curves = os.path.join(BASE_DIR, "lstm_plots", "lstm_training_curves.png")
if os.path.exists(lstm_curves):
    import shutil
    shutil.copy(lstm_curves, os.path.join(PLOTS_DIR, 'fig5_lstm_training_curves.png'))
    print("Figure 5 copied from lstm_plots.")

# Figure 6: Performance in High-Volatility Periods
fig, ax = plt.subplots(figsize=(16, 5))
rolling_vol = pd.Series(actual_ret).rolling(21).std()
sq_err_garch = garch_err**2
sq_err_lstm  = lstm_err**2
rolling_se_garch = sq_err_garch.rolling(21).mean()
rolling_se_lstm  = sq_err_lstm.rolling(21).mean()

ax.plot(dates, rolling_se_garch, color='#1565C0', linewidth=1.0, label='GARCH 21-d Rolling MSE')
ax.plot(dates, rolling_se_lstm, color='#E65100', linewidth=1.0, label='LSTM 21-d Rolling MSE')
ax2 = ax.twinx()
ax2.fill_between(dates, 0, rolling_vol, alpha=0.15, color='gray', label='21-d Rolling Ïƒ')
ax2.set_ylabel('Rolling Volatility', color='gray')
ax.set_title('Figure 6: Forecast Accuracy vs Market Volatility')
ax.set_ylabel('Rolling MSE')
ax.set_xlabel('Date')
ax.legend(loc='upper left')
fig.savefig(os.path.join(PLOTS_DIR, 'fig6_high_vol_performance.png'))
plt.close()

print(f"All figures saved to: {PLOTS_DIR}/")

# â”€â”€ MARKDOWN REPORT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
md = []
md.append("# Final Research Summary")
md.append("")
md.append("## Project")
md.append("*Comparaison des modÃ¨les statistiques et d'apprentissage automatique")
md.append("pour la prÃ©vision financiÃ¨re (ARIMA-GARCH et LSTM)*")
md.append("")
md.append(f"**Evaluation Period:** {dates.iloc[0].date()} â†’ {dates.iloc[-1].date()}")
md.append(f"**Evaluation Observations:** {len(eval_df)}")
md.append("")

# Table 1
if table1_data:
    md.append("### Table 1: Statistical Model Comparison")
    t1 = pd.DataFrame(table1_data)
    md.append(t1.to_markdown(index=False))
    md.append("")

# Table 2
if os.path.exists(HYPER_CSV):
    md.append("### Table 2: LSTM Configurations")
    t2 = pd.read_csv(HYPER_CSV)
    md.append(t2.to_markdown(index=False))
    md.append("")

# Tables 3-5
for task_name, table_num in [('Return', 3), ('Price', 4), ('Volatility', 5)]:
    subset = metrics_df[metrics_df['Task'] == task_name]
    if len(subset) > 0:
        md.append(f"### Table {table_num}: {task_name} Forecasting Performance")
        md.append(subset.drop(columns=['Task']).to_markdown(index=False))
        md.append("")

# Table 6
md.append("### Table 6: Diebold-Mariano Tests")
md.append(dm_df.to_markdown(index=False))
md.append("")

# Conclusions
md.append("---")
md.append("")
md.append("## Research Conclusions")
md.append("")
md.append("### Summary")
md.append("The above tables and figures provide the empirical evidence needed to answer")
md.append("the core research question. Review the DM test p-values (Table 6) to determine")
md.append("whether any observed metric differences are statistically significant at Î±=0.05.")
md.append("")
md.append("### Methodological Notes")
md.append("- All GARCH parameters estimated on Training set only (2000â€“2018)")
md.append("- LSTM scalers fitted on Training set only")
md.append("- No hyperparameter tuning on the Test set")
md.append("- Chronological (non-shuffled) data split")
md.append("- No outlier removal or data winsorization")
md.append("- Reproducible random seed = 42")
md.append("")
md.append("### Figures")
md.append("| # | Description |")
md.append("|---|-------------|")
md.append("| 1 | Actual vs predicted returns |")
md.append("| 2 | Actual vs predicted prices |")
md.append("| 3 | Actual vs predicted volatility |")
md.append("| 4 | Forecast errors over time |")
md.append("| 5 | LSTM training/validation loss curves |")
md.append("| 6 | Forecast accuracy during high-volatility periods |")

with open(REPORT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))

print(f"\nFinal report saved to: {REPORT_MD}")
print("\nResults and figures generation complete.")
