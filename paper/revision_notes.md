# Scientific Paper Revision Notes (Final Polish)

The manuscript (`main.tex`) and bibliography (`references.bib`) have been updated with the final set of targeted scientific and presentation corrections.

Below is the summary of the final substantive changes:

1. **Author Block Added:** Included the author details (Majd Souissi, Department / University, majdsouissi625@gmail.com) prominently below the title.
2. **LSTM Terminology:** Replaced instances of "three distinct LSTM architectures" with the more technically accurate "three distinct LSTM forecasting configurations."
3. **Parametric Efficiency:** Reworded the discussion regarding parametric efficiency. Instead of claiming the LSTM "fails to match this parametric efficiency", the text now objectively states: "The LSTM, tasked with discovering these dynamics from a 60-day window, does not achieve comparable forecasting accuracy under squared-error loss in this experiment."
4. **Volatile Periods Analysis:** Toned down the causal/inferential language regarding market stress regimes. "Predictive breakdown is tightly correlated with regime shifts" was replaced with the descriptive observation: "Higher forecast errors are visually concentrated around periods of elevated market volatility."
5. **Multiple DM Comparisons:** Added an explicit disclosure in the Diebold-Mariano methodology section (3.8) stating: "The DM comparisons are reported at the nominal 5% significance level; no multiple-comparison adjustment was applied."
6. **Abstract Wording:** Refined the QLIKE conclusion in the abstract from "produces a numerically lower score" to the cleaner "attains a lower QLIKE score."
7. **Title Retained:** Kept the title "Forecasting Daily S&P 500 Returns and Volatility: An Empirical Comparison of GJR-GARCH and LSTM Networks" as it perfectly reflects the empirical study.
8. **Student-t Clarification:** Added a sentence explicitly connecting the degrees of freedom estimate ($\nu \approx 7.14$) to the dataset's heavy tails: "Furthermore, the estimated degrees of freedom ($\nu \approx 7.14$) indicate substantial tail thickness relative to a Gaussian innovation assumption."
9. **Recent Literature Expanded:** Addressed the thinness of the Related Work section by finding and citing three relevant empirical comparisons published between 2022 and 2024:
   - Petrozziello (2022) - *Soft Computing*
   - Cai et al. (2023) - *BCP Business & Management*
   - Xu et al. (2024) - *Journal of Computer Science and Software Applications*
   These citations were integrated into a new paragraph detailing that while LSTMs excel at non-linear patterns, they do not universally outclass econometrics, which has led to the current state-of-the-art trend of hybrid GARCH-LSTM frameworks.
