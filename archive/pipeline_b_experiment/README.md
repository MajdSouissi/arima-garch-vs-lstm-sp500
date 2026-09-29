# Pipeline B (Alternative Experiment)

This directory contains `run_experiment.py` and its outputs (`results_v2/`).

**IMPORTANT NOTE:**
This pipeline was an earlier, alternative experiment. It is **NOT** the canonical methodology used in the final published thesis or paper. 

### Methodological Differences from Canonical Pipeline A:
1. **GARCH Specification**: Uses a **Symmetric GARCH(1,1)** (no leverage term), whereas the canonical pipeline uses an asymmetric GJR-GARCH(1,1).
2. **Mean Specification**: Uses a two-step ARIMA(2,0,0) process selected via BIC, whereas the canonical pipeline fits Constant and AR(1) models jointly with the volatility equation.
3. **LSTM Configuration**: Trains 2 LSTM models instead of the 3 configurations explored in the final paper.
4. **Results**: The outcomes differ significantly from the published thesis, particularly regarding the Diebold-Mariano test results for volatility forecasting under the QLIKE loss function.

This code is preserved here for historical context and reproducibility of the early experimental phases, but should not be considered the definitive analysis of the project.
