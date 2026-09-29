# ARIMA-GARCH vs LSTM for Financial Forecasting

## 1. Project Title
Comparaison des modèles statistiques et d'apprentissage automatique pour la prévision financière (ARIMA-GARCH et LSTM)

## 2. Research Question
How do traditional statistical models (ARIMA-GARCH) compare with modern deep learning architectures (LSTM) in forecasting daily financial returns and volatility, particularly during periods of market stress?

## 3. Motivation
Accurate financial forecasting is central to risk management and portfolio optimization. While GARCH models are the classical econometric standard for modeling volatility clustering, Recurrent Neural Networks (like LSTMs) offer non-linear, data-driven alternatives. This project aims to systematically evaluate and compare their predictive accuracy in out-of-sample settings.

## 4. Dataset and Sources
The analysis is conducted on the **S&P 500 index**.
- **Historical Data**: Daily closing prices.
- **Source**: Publicly available financial data (e.g., Investing.com / FRED).
- **Timeframe**: August 2000 to August 2026.

## 5. Return and Volatility Definitions
- **Returns**: Daily log returns computed as $r_t = \ln(P_t / P_{t-1})$.
- **Volatility Proxy**: Realized squared returns ($r_t^2$) or absolute returns ($|r_t|$) are used as proxies for unobservable daily conditional variance.

## 6. Canonical Methodology
This repository contains the **canonical Pipeline A** analysis as presented in the final Master's thesis. 
The pipeline strictly enforces out-of-sample evaluation with no look-ahead bias, fixed-parameter forecasting for statistical models, and causal feature engineering for machine learning models.

The methodology proceeds in 9 steps (found in `src/`):
1. Data Preparation
2. Statistical Diagnosis
3. Mean & Variance Investigation
4. GARCH Modeling
5. LSTM Data Preparation
6. LSTM Modeling
7. Forecasting Evaluation
8. Statistical Comparison (DM Tests)
9. Results and Figures Generation

## 7. GARCH Specification
The statistical baseline relies on the **GJR-GARCH(1,1)** model with Student-t distributed innovations to capture volatility clustering, asymmetry (leverage effects), and fat tails. 
Two mean specifications are tested:
- Constant Mean
- AR(1) Mean

## 8. LSTM Specification
Three parsimonious LSTM configurations were trained using PyTorch:
- **LSTM_Base**: Predicts next-day return using only past returns.
- **LSTM_Enhanced**: Predicts next-day return using returns, squared returns, and rolling volatility/MAE.
- **LSTM_Volatility**: Predicts next-day squared return (volatility proxy) using the enhanced feature set.

## 9. Chronological Split
The dataset is chronologically split to prevent data leakage:
- **Training**: 2000-08-15 to 2018-12-31
- **Validation**: 2019-01-02 to 2022-12-30 (Used for LSTM early stopping)
- **Test (Out-of-Sample)**: 2023-01-03 to 2026-08-13 (906 observations)

## 10. Evaluation Metrics
- **Returns**: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Directional Accuracy (DirAcc)
- **Prices**: RMSE, MAE, Mean Absolute Percentage Error (MAPE)
- **Volatility**: RMSE, MAE, Quasi-Likelihood (QLIKE) loss

## 11. Diebold-Mariano Test
The statistical significance of differences in predictive accuracy between models is evaluated using the two-sided **Diebold-Mariano (DM) test**.

## 12. Main Results
The canonical results indicate that the choice of model heavily depends on the forecasting task:
- **Return Forecasting**: The GARCH models with Constant or AR(1) mean specifications performed marginally better than the LSTM models, showing statistically significant improvements under Absolute Error (AE) loss. Directional accuracy hovered around ~50-56% for all models.
- **Volatility Forecasting**: The GJR-GARCH(1,1) model and the LSTM_Volatility model achieved comparable performance under the QLIKE loss function (no statistically significant difference). However, under Squared Error (SE) loss, the GJR-GARCH model proved to be significantly more accurate.
- **Conclusion**: The analysis does not establish universal superiority for either family of models. GARCH remains a highly competitive, robust benchmark for financial time series, while LSTMs show comparable accuracy in specific volatility contexts but offer no dominant advantage in this specific univariate configuration.

*Full detailed tables and metrics are available in `results/reports/09_final_research_summary.md`.*

## 13. Repository Structure
- `data/`: Raw and processed datasets.
- `src/`: Core Python pipeline scripts (01 to 09).
- `scripts/`: Supplementary analysis and plotting scripts.
- `results/`: Forecast outputs, evaluation metrics, and summary reports.
- `figures/`: EDA and final publication figures.
- `models/`: Saved PyTorch model weights (`.pt`).
- `archive/`: Deprecated or alternative experiments.

## 14. Installation
```bash
git clone https://github.com/username/arima-garch-vs-lstm-sp500.git
cd arima-garch-vs-lstm-sp500
pip install -r requirements.txt
```

## 15. Reproduction Instructions
To reproduce the canonical analysis, execute the scripts in `src/` sequentially from `01_data_preparation.py` to `09_results_and_figures.py`. 
*Note: Due to the stochastic nature of GPU/CPU deep learning training, exact LSTM weights may vary slightly across different hardware setups despite the fixed random seed (42).*

## 16. Limitations
- The analysis is purely univariate (using only price-derived features).
- The test period is fixed and does not employ a rolling-window re-estimation strategy, which might improve the statistical models' adaptiveness.

## 17. Citation
If you use this code or methodology, please cite the associated Master's thesis.
