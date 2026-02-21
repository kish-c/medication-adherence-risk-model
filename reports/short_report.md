# Short Technical Report — Medication Adherence Risk Model

## Problem

Medication non-adherence is one of the more tractable problems in population
health analytics. The signal is there in refill data — patients who are drifting
toward a gap tend to show it weeks before the gap opens. The question is whether
a simple ML model can pick that up early enough to be actionable.

This project tests that hypothesis on synthetic pharmacy claims data.

---

## Approach

The pipeline generates a synthetic patient population (n=3,000) with realistic
refill behavior, engineers adherence features from refill history, and trains two
classifiers to predict 90-day non-adherence.

**Label definition:** A patient is labeled non-adherent (`non_adherent_90d = 1`)
if their Proportion of Days Covered (PDC) in the final 90-day observation window
falls below 0.80 — the standard NCQA/CMS threshold used in Star Ratings programs.

**Class distribution:** 39.2% non-adherent, 60.8% adherent.

---

## Feature Engineering

The most informative features were adherence history metrics computed over a
prior 180-day window:

- **PDC (prior_pdc):** Were they covered for most days before the prediction window?
- **MPR (prior_mpr):** Total days supply relative to window — captures stockpiling behavior.
- **avg_refill_gap_days:** Mean days between consecutive fills. The single strongest predictor.
- **max_refill_gap_days:** Worst-case gap in history. Captures one-time disruptions.
- **refill_gap_std:** Consistency of refill timing — irregular fillers tend to eventually lapse.

SQL window functions implement PDC/MPR logic in `sql/feature_engineering.sql`.
The same logic runs in Python via `src/feature_engineering.py`.

---

## Results

Both models performed strongly, which is somewhat expected given the synthetic
data has clean signal. The more interesting finding is the agreement between models
on feature importance.

| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|----|
| Logistic Regression | 0.976 | 0.886 | 0.894 | 0.890 |
| Random Forest | 0.979 | 0.849 | 0.911 | 0.879 |

The logistic regression is the recommended model for deployment because:
1. Coefficients are interpretable and auditable
2. Probability scores are easier to explain to clinical stakeholders
3. Performance difference vs. Random Forest is marginal (0.003 AUC)

**Top predictors (both models agree):**
1. `prior_pdc` — strongest negative predictor of non-adherence
2. `avg_refill_gap_days` — strongest positive predictor
3. `prior_mpr` — second-strongest protective factor
4. `max_refill_gap_days` — history of a bad gap predicts future gaps

---

## What I'd Do With More Time

**Calibration.** The model's probabilities aren't calibrated — a score of 0.7
doesn't mean 70% chance of non-adherence. Platt scaling or isotonic regression
would fix this and make threshold-setting more principled.

**Threshold optimization.** Right now we use 0.5. In practice, the cost of a
false negative (missing a patient who lapses) is likely much higher than the cost
of a false positive (unnecessary outreach). A proper cost-sensitive threshold
would be set based on the ratio of those costs.

**Fairness checks.** Before any deployment, I'd want to know if recall is
significantly lower for any demographic subgroup. Differential performance is
common in health models and needs to be caught before the model makes it into
a care coordinator's queue.

**Richer features.** Real claims data would include diagnosis codes, ER visits,
lab values, and insurance type — all of which have documented relationships with
adherence. The current feature set is pharmacy-data-only.

**Time series modeling.** The current model treats each patient as a static
snapshot. A sequential model (LSTM, or even just rolling lag features) would
capture whether adherence is trending up or down, which is arguably more
actionable than a point-in-time score.

---

*All data is synthetic. No PHI was used. Not for clinical deployment.*
