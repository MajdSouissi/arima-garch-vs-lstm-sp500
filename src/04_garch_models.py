"""
=============================================================================
04_garch_models.py
Final Statistical Benchmark (ARIMA-GARCH)
=============================================================================
Fits the selected baseline and selected conditional mean + GJR-GARCH(1,1)
with Student-t innovations on the training set.
Generates 1-step-ahead out-of-sample forecasts for the Test set.
=============================================================================
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

warnings.filterwarnings('ignore')

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR = str(Path(__file__).resolve().parent.parent)
INPUT_CSV = os.path.join(BASE_DIR, "SP500_master_2000_2026.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "garch_forecasts.csv")
REPORT_PATH = os.path.join(BASE_DIR, "04_garch_report.txt")

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

# â”€â”€ 1. Data Loading & Splitting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df = pd.read_csv(INPUT_CSV, parse_dates=['Date'])
# Ensure we drop the first NaN row
df = df.dropna(subset=['Log_Return']).reset_index(drop=True)

# Hardcoded split dates
train_end_dt = pd.to_datetime('2018-12-31')
val_end_dt   = pd.to_datetime('2022-12-31')

train_df = df[df['Date'] <= train_end_dt]
val_df   = df[(df['Date'] > train_end_dt) & (df['Date'] <= val_end_dt)]
test_df  = df[df['Date'] > val_end_dt]

train_size = len(train_df)
val_size   = len(val_df)
test_size  = len(test_df)

train_end_idx = train_size - 1
val_end_idx   = train_size + val_size - 1

train_end_date = train_df['Date'].iloc[-1]
val_end_date   = val_df['Date'].iloc[-1]
test_start_date = test_df['Date'].iloc[0]

returns = pd.Series(df['Log_Return'].values, index=df['Date'])
dates = df['Date']

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    log("=============================================================================", f)
    log(" DATA SPLITTING", f)
    log("=============================================================================", f)
    log(f"Total Observations: {len(df)}", f)
    log(f"Train Period: {train_df['Date'].iloc[0].date()} to {train_end_date.date()} ({train_size} obs)", f)
    log(f"Val Period:   {val_df['Date'].iloc[0].date()} to {val_end_date.date()} ({val_size} obs)", f)
    log(f"Test Period:  {test_start_date.date()} to {test_df['Date'].iloc[-1].date()} ({test_size} obs)", f)
    log("", f)

    # â”€â”€ 2. Model 1: Constant Mean + GJR-GARCH(1,1) + Student-t â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log("=============================================================================", f)
    log(" MODEL 1: Constant Mean + GJR-GARCH(1,1) + Student-t", f)
    log("=============================================================================", f)
    
    am1 = arch_model(returns, mean='Constant', vol='Garch', p=1, o=1, q=1, dist='t')
    # Fit only on the training set
    res1 = am1.fit(last_obs=train_end_date, disp='off')
    
    log(res1.summary().as_text(), f)
    log("", f)
    
    # â”€â”€ 3. Model 2: AR(1) + GJR-GARCH(1,1) + Student-t â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log("=============================================================================", f)
    log(" MODEL 2: AR(1) + GJR-GARCH(1,1) + Student-t", f)
    log("=============================================================================", f)
    
    # MA(1) is modeled as mean='ARX' with no AR lags, or simply using ARMA in arch package
    # arch_model mean='AR' with lags=0 and vol='Garch' doesn't do MA.
    # Actually, arch_model supports mean='ARMA', but wait, ARMA mean is deprecated in favor of ARX.
    # To keep it rigorous and within standard arch capabilities, we will use mean='AR', lags=1.
    # Previous tests showed ARMA(0,1) or AR(1) had similar BIC. We will use AR(1) here for simplicity 
    # since `arch` natively supports it beautifully with mean='AR', lags=1.
    am2 = arch_model(returns, mean='AR', lags=1, vol='Garch', p=1, o=1, q=1, dist='t')
    res2 = am2.fit(last_obs=train_end_date, disp='off')
    
    log(res2.summary().as_text(), f)
    log("", f)

    # â”€â”€ 4. Diagnostics on Training Residuals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log("=============================================================================", f)
    log(" IN-SAMPLE DIAGNOSTICS (Model 2)", f)
    log("=============================================================================", f)
    
    # Get standardized residuals for the training period
    std_resid2 = (res2.resid / res2.conditional_volatility).dropna()
    std_resid2_train = std_resid2.loc[:train_end_date]
    
    # Ljung-Box on std residuals
    lb_z = acorr_ljungbox(std_resid2_train, lags=[10, 20], return_df=True)
    log("Ljung-Box on Std Residuals (z_t):", f)
    log(lb_z.to_string(), f)
    
    # Ljung-Box on squared std residuals
    lb_z2 = acorr_ljungbox(std_resid2_train**2, lags=[10, 20], return_df=True)
    log("\nLjung-Box on Squared Std Residuals (z_t^2):", f)
    log(lb_z2.to_string(), f)
    
    # ARCH-LM
    arch_lm = het_arch(std_resid2_train, nlags=10)
    log(f"\nARCH-LM(10) on z_t: LM-stat={arch_lm[0]:.4f}, p-value={arch_lm[1]:.4e}", f)
    log("", f)

    # â”€â”€ 5. Out-of-Sample Forecasting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log("Generating 1-step-ahead out-of-sample forecasts...", f)
    
    # The forecast() method with align='target' produces forecasts for t+1
    forecasts1 = res1.forecast(horizon=1, align='target')
    forecasts2 = res2.forecast(horizon=1, align='target')
    
    # Extract predicted mean and variance
    # The index of forecasts.mean matches the target date
    pred_mean_1 = forecasts1.mean['h.1']
    pred_var_1  = forecasts1.variance['h.1']
    
    pred_mean_2 = forecasts2.mean['h.1']
    pred_var_2  = forecasts2.variance['h.1']
    
    forecast_df = df[['Date', 'Close', 'Log_Return']].copy()
    forecast_df['Split'] = 'Train'
    forecast_df.loc[val_end_idx - val_size + 1 : val_end_idx, 'Split'] = 'Val'
    forecast_df.loc[val_end_idx + 1 :, 'Split'] = 'Test'
    
    # Set index to match the DatetimeIndex of the predictions for automatic alignment
    forecast_df.set_index('Date', inplace=True)
    
    forecast_df['GARCH_M1_Mean'] = pred_mean_1
    forecast_df['GARCH_M1_Vol']  = np.sqrt(pred_var_1)
    
    forecast_df['GARCH_M2_Mean'] = pred_mean_2
    forecast_df['GARCH_M2_Vol']  = np.sqrt(pred_var_2)
    
    # Reset index for saving
    forecast_df.reset_index(inplace=True)
    
    # Save to CSV
    forecast_df.to_csv(OUTPUT_CSV, index=False)
    log(f"Saved forecasts to: {OUTPUT_CSV}", f)
    
print("GARCH modeling complete. Check 04_garch_report.txt and garch_forecasts.csv.")
