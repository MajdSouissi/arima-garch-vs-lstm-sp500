# Final Scientific Revision Notes

The manuscript (`main.tex`) and bibliography (`references.bib`) have undergone a final rigorous scientific polish to ensure precise literature positioning, exact terminology, and absolute adherence to the empirical findings without overclaiming.

## Substantive Changes:

1. **Related Work Expansion:** 
   - Added a detailed paragraph incorporating 5 recent (2022–2025) empirical studies: Petrozziello (2022), Cai et al. (2023), Mualifah et al. (2024), Xu et al. (2024), and Li et al. (2025). 
   - Briefly identified their focus (e.g., S&P 500, Indonesia Stock Exchange, hybrid architectures) and main conclusions.
   - Explicitly differentiated the present study by noting its focus on a strictly chronological one-step-ahead test period, formal Diebold-Mariano testing under multiple losses, and evaluation of both return and conditional variance.

2. **Introduction & Terminology Adjustments:**
   - Replaced "structural efficiency" with "structured econometric representation" to avoid implying formal statistical efficiency.
   - Clarified the volatility introduction to state that "the conditional variance exhibits persistent dependence that can be modeled statistically," rather than claiming it is universally "predictable."
   - Refined the "intraday shock" sentence to precisely state: "Unexpected shocks occurring during the forecasted trading day cannot be anticipated from information available at the previous close."

3. **QLIKE Discussion Precision:**
   - Replaced claims about LSTM being a "viable alternative" with the strict statistical finding: "Under QLIKE, the LSTM is statistically indistinguishable from the GARCH benchmarks in this experiment." Emphasized that non-significance does not equal practical dominance.

4. **LSTM Configuration Table:**
   - Added a compact, precise table (Table 2) detailing the exact final LSTM settings: 60-day lookback, 1 hidden layer, 32 units, 0.2 dropout, Adam optimizer (LR=0.001), MSE loss, and early stopping (patience=10 epochs).

5. **Volatility Target Clarification:**
   - Added a concise methodological paragraph in Section 3.6 clarifying exactly what is forecasted and evaluated: GARCH forecasts the conditional variance $\sigma_{t+1}^2$, the LSTM predicts $r_{t+1}^2$, and both are evaluated against $r_{t+1}^2$. It explicitly names $r_{t+1}^2$ as a "noisy observable variance proxy" rather than true realized variance.

6. **Model Terminology Standardization:**
   - Consistently applied the labels: *AR(1)-GJR-GARCH(1,1)-Student-t*, *LSTM Base Return*, *LSTM Enhanced Return*, and *LSTM Volatility*.
   - Replaced "LSTM architectures" with "LSTM forecasting configurations" since the underlying neural architecture is identical.

7. **Restrained Discussion & Conclusion:**
   - Removed implications of universal superiority or generalization. 
   - Confined the conclusions specifically to the S&P 500 dataset, daily frequency, one-step-ahead forecasting, and the tested loss functions.

8. **Added Limitation:**
   - Explicitly acknowledged that "The comparative conclusions may also depend on the specific feature set and LSTM configuration adopted here."

9. **Writing Style & Tone:**
   - Purged all inflated adjectives (e.g., "profound", "definitive", "elegant", "formidable").
   - Adopted a strictly observational and objective tone ("we observe", "the evidence indicates").

10. **Scientific Consistency Audit:**
    - Verified that all RMSE/MAE/QLIKE numbers, DM $p$-values, dates (2023-01-03 to 2026-08-13), sample sizes (906), and model specifications strictly match the Python pipeline outputs. No numerical values were altered.
