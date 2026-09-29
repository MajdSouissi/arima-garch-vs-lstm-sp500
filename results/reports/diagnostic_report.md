# S&P 500 — Statistical Diagnosis Report

**Master's Research Project:**  
*Comparaison des modèles statistiques et d'apprentissage automatique
pour la prévision financière (ARIMA-GARCH et LSTM)*

**Date generated:** 2026-08-14 17:18
**Dataset:** SP500_master_2000_2026.csv
**Observations:** 6538
**Date range:** 2000-08-14 → 2026-08-13

---

## Summary of Statistical Tests

| Category | Test | Statistic | p-value | Conclusion (α=0.05) |
|----------|------|-----------|---------|---------------------|
| Stationarity | ADF — Close | 0.4840 | 0.996821 | FAIL TO REJECT H0 → non-stationary (expected for prices) |
| Stationarity | KPSS — Close | 2.5712 | 0.0100 | REJECT H0 → non-stationary |
| Stationarity | ADF — Log_Return | -20.1230 | 0.000000 | REJECT H0 → stationary (expected for returns) |
| Stationarity | KPSS — Log_Return | 0.473892 | 0.0475 | REJECT H0 → non-stationary (unexpected) |
| Distribution | Skewness | -0.3519 | — | Negative skew → left tail heavier |
| Distribution | Excess Kurtosis | 11.0425 | — | Leptokurtic (heavy tails) |
| Distribution | Jarque-Bera | 33291.18 | 0.00e+00 | REJECT H0 → returns are NOT normally distributed |
| Linear dependence | Ljung-Box (returns, lag=5) | 75.20 | 0.0000 | REJECT H0 → serial correlation |
| Linear dependence | Ljung-Box (returns, lag=10) | 101.81 | 0.0000 | REJECT H0 → serial correlation |
| Linear dependence | Ljung-Box (returns, lag=20) | 165.48 | 0.0000 | REJECT H0 → serial correlation |
| Linear dependence | Ljung-Box (returns, lag=40) | 199.63 | 0.0000 | REJECT H0 → serial correlation |
| Volatility dependence | Ljung-Box (r², lag=5) | 3566.62 | 0.00e+00 | REJECT H0 → volatility clustering |
| Volatility dependence | Ljung-Box (r², lag=10) | 6111.38 | 0.00e+00 | REJECT H0 → volatility clustering |
| Volatility dependence | Ljung-Box (r², lag=20) | 8617.74 | 0.00e+00 | REJECT H0 → volatility clustering |
| Volatility dependence | Ljung-Box (r², lag=40) | 10573.35 | 0.00e+00 | REJECT H0 → volatility clustering |
| ARCH effects | ARCH-LM (lag=1) | 650.00 | 2.24e-143 | REJECT H0 → ARCH effects |
| ARCH effects | ARCH-LM (lag=5) | 1656.98 | 0.00e+00 | REJECT H0 → ARCH effects |
| ARCH effects | ARCH-LM (lag=10) | 1767.21 | 0.00e+00 | REJECT H0 → ARCH effects |
| ARCH effects | ARCH-LM (lag=20) | 1873.32 | 0.00e+00 | REJECT H0 → ARCH effects |
| ARCH effects | ARCH-LM (lag=40) | 1945.29 | 0.00e+00 | REJECT H0 → ARCH effects |

---

## Key Distribution Statistics

| Statistic | Value |
|-----------|-------|
| N | 6537 |
| Mean | 0.025304 |
| Std Dev | 1.208061 |
| Skewness | -0.3519 |
| Excess Kurtosis | 11.0425 |
| Min | -12.7652 |
| Max | 10.9644 |
| Jarque-Bera (p) | 0.00e+00 |

---

## Research Conclusions

### A. Is the price series stationary?
**No.** ADF fails to reject H0 (stat=0.4840, p=0.9968); KPSS rejects H0 (stat=2.5712, p=0.0100). Both tests consistently indicate non-stationarity. The price series exhibits I(1) behavior (unit root).

### B. Are log returns stationary?
**Largely yes (in mean), with a nuance.** ADF strongly rejects the unit-root null (stat=−20.1230, p≈0.000), providing strong evidence against a unit root. However, KPSS *borderline* rejects level-stationarity (stat=0.4739, p=0.0475) at the 5% level. Taken together, the ADF evidence against a unit root is overwhelming, while the KPSS borderline rejection may reflect slowly evolving unconditional variance (volatility clustering) rather than non-stationarity in the mean. The return series is treated as mean-stationary for ARMA modeling, but the variance is clearly NOT constant over time.

### C. Are returns approximately normally distributed?
**No.** Jarque-Bera decisively rejects normality (JB=33291.18, p=0.00e+00). The distribution is leptokurtic with negative skew.

### D. Is there evidence of heavy tails?
**Yes.** Excess kurtosis = 11.04 (normal = 0). The Q-Q plot shows systematic departure from normality in both tails. Extreme returns occur more frequently than a Gaussian model predicts.

### E. Is there meaningful linear autocorrelation?
**Weak.** 12/40 ACF lags are individually significant. Ljung-Box at lag 20: p=0.0000. Linear dependence in returns is present but small in magnitude, consistent with near-efficient markets.

### F. Is there evidence of volatility clustering?
**Yes, very strong.** 40/40 ACF lags of squared returns and 40/40 ACF lags of absolute returns are significant. Ljung-Box on r² is highly significant at all lags tested. This is the canonical signature of volatility clustering.

### G. Is there statistically significant ARCH behavior?
**Yes.** Engle's ARCH-LM test rejects H0 at all tested lags (p ≈ 0 at every lag). Conditional heteroskedasticity is strongly supported by the data.

### H. Do these results motivate an ARIMA-GARCH model?
**Yes.** The empirical evidence supports the following modeling rationale:

1. **ARIMA component:** Log returns are stationary and show weak but detectable linear autocorrelation, justifying a low-order ARMA model for the conditional mean.
2. **GARCH component:** Strong ARCH effects and volatility clustering indicate that the conditional variance is time-varying and predictable, justifying a GARCH-type specification for the conditional variance.
3. **Non-normality:** Heavy tails suggest that a Student-t or skewed-t innovation distribution may be preferable to Gaussian innovations in the GARCH specification.
4. **LSTM comparison:** The presence of complex, potentially nonlinear patterns in both returns and volatility provides a rationale for evaluating whether deep learning (LSTM) can capture dynamics that parametric ARIMA-GARCH models may miss.

---

## Figures

| Figure | Description |
|--------|-------------|
| fig3_distribution.png | Histogram and Q-Q plot of log returns |
| fig4_acf_pacf_returns.png | ACF and PACF of log returns |
| fig5_acf_pacf_volatility.png | ACF/PACF of squared and absolute returns |
| fig7a_returns_timeseries.png | Log returns over time |
| fig7b_squared_returns.png | Squared returns over time |
| fig7c_volatility_panel.png | Combined volatility panel (4 subplots) |

---
*End of diagnostic report.*