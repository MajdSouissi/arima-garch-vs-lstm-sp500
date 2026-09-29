# GARCH(1,1) Symmetric — Out-of-Sample Evaluation Report

**Generated:** 2026-09-05 10:25

## 1. Data

| Item | Value |
|------|-------|
| Dataset | S&P 500 daily log returns (%) |
| Source | `SP500_master_2000_2026.csv` |
| Total observations | 6537 |
| Train | 2000-08-15 → 2018-12-31 (4623 obs) |
| Validation | 2019-01-02 → 2022-12-30 (1008 obs) |
| Test | 2023-01-03 → 2026-08-13 (906 obs) |
| Evaluation observations | 906 (inner join with LSTM dates) |

## 2. Model Specification

```
Conditional Mean:   r_t = μ + ε_t          (Constant Mean)
Conditional Var:    σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}   (GARCH(1,1) symmetric)
Innovations:        z_t ~ Student-t(ν)

This is a SYMMETRIC GARCH — no gamma/leverage parameter.
arch_model(..., vol='Garch', p=1, o=0, q=1, dist='t')
```

### Specification Choices

| Choice | Justification |
|--------|---------------|
| Constant Mean | Consistent with GARCH_Const (GJR Model 1); `mean_variance_report.md` finds 'constant-mean is defensible' |
| Student-t | Consistent with GJR-GARCH models; motivated by excess kurtosis ≈ 11 |
| GARCH(1,1) symmetric | Requested as baseline comparison for LSTM_Vol |

## 3. Estimated Parameters (Training Set Only)

| Parameter | Value |
|-----------|-------|
| μ (mean) | 0.064898 |
| ω (intercept) | 0.009869 |
| α (ARCH) | 0.107344 |
| β (GARCH) | 0.892115 |
| α + β (persistence) | 0.999459 |
| ν (Student-t df) | 6.2855 |
| AIC | 12327.62 |
| BIC | 12359.82 |
| Log-Likelihood | -6158.81 |
| Estimation obs | 4623 |

## 4. OOS Forecast Protocol

- **Horizon:** 1-step-ahead
- **Method:** `res.forecast(horizon=1, align='target')`
- **Parameters fixed** at training-set estimates (no re-estimation)
- **No test data** used in parameter estimation

## 5. Volatility Evaluation Metrics (Test Period)

Target: `Realized_Vol_Proxy` = r²_t (squared log return)

| Model | RMSE | MAE | QLIKE |
|-------|------|-----|-------|
| GARCH(1,1) symmetric | 3.254202 | 1.017389 | 1.653101 |
| LSTM_Vol | 3.203975 | 1.004294 | 1.577264 |

## 6. Return Evaluation Metrics (Test Period)

| Model | RMSE | MAE | DirAcc (%) |
|-------|------|-----|------------|
| GARCH(1,1) symmetric | 0.932608 | 0.653527 | 56.18 |
| LSTM_Base | 0.930757 | 0.661675 | 49.45 |

## 7. Diebold-Mariano Tests

| Task       | Model_1       | Model_2   | Loss   |   DM_Stat |   p_value |   N | Conclusion                                    |
|:-----------|:--------------|:----------|:-------|----------:|----------:|----:|:----------------------------------------------|
| Volatility | GARCH11_Const | LSTM_Vol  | QLIKE  |    3.677  |    0.0002 | 906 | LSTM_Vol significantly better (p=0.0002)      |
| Volatility | GARCH11_Const | LSTM_Vol  | SE     |    0.5879 |    0.5566 | 906 | No significant difference (p=0.5566)          |
| Return     | GARCH11_Const | LSTM_Base | SE     |    0.1511 |    0.8799 | 906 | No significant difference (p=0.8799)          |
| Return     | GARCH11_Const | LSTM_Base | AE     |   -2.1787 |    0.0294 | 906 | GARCH11_Const significantly better (p=0.0294) |

## 8. COVID-19 Stress Analysis

A stress figure was produced for the COVID-19 period (Feb–May 2020).
This period falls in the **validation set**, not the test set.

Key observations:
- LSTM_Vol forecasts are **unavailable** for this period (LSTM test begins 2023)
- The figure compares GARCH(1,1) symmetric vs GJR-GARCH(1,1) vs realized |r_t|
- The purpose is to observe model behavior during extreme volatility regimes
- This is NOT a formal evaluation — the official test period remains 2023–2026

Figure: `fig_covid_stress_volatility.png`

## 9. Hypotheses and Limitations

1. **Fixed parameters:** GARCH parameters estimated once on training data, not re-estimated
   during the test period. This is consistent with the GJR-GARCH protocol.
2. **No rolling/expanding window:** Same protocol as existing models.
3. **Constant Mean chosen** based on existing project conventions. AR(1) was also
   tested in the GJR setup but is not reproduced here for the symmetric GARCH.
4. **Student-t chosen** based on existing project conventions. The `mean_variance_report.md`
   confirms Student-t improves fit over normal for all GARCH variants.
