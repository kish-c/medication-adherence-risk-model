# Model Card — Medication Non-Adherence Risk Model

## Overview

This model predicts the likelihood that a patient will be non-adherent to their
medication regimen within a rolling 90-day observation window. It is designed to
surface high-risk patients for proactive outreach, not to make clinical decisions.

---

## Intended Use

**Primary use case:** Identify patients who may benefit from pharmacist or care
coordinator outreach before a refill gap widens into a clinical event.

**Intended users:** Data analysts, clinical operations teams, population health managers.

**Out-of-scope uses:**
- Direct clinical decision-making without human review
- Determining treatment plans or medication changes
- Any use on real patient data without IRB/compliance approval

---

## Data

All data used to train and evaluate this model is **fully synthetic**. No real
patient data, PHI, or de-identified records were used at any stage.

The synthetic dataset was generated to mimic realistic pharmacy claims behavior:
- 3,000 simulated patients with age and chronic disease flags
- 6,598 prescription records across 5 drug classes
- 29,065 refill events with realistic gap distributions

**Key features:**

| Feature | Description |
|---------|-------------|
| `prior_pdc` | Proportion of Days Covered in prior 180-day window |
| `prior_mpr` | Medication Possession Ratio (capped at 1.0) |
| `avg_refill_gap_days` | Mean days between refills |
| `max_refill_gap_days` | Longest observed gap |
| `refill_gap_std` | Variability in gap length |
| `switch_count` | Number of medication class switches |
| `overlap_days` | Total early-refill overlap days |
| `chronic_flag` | Binary indicator of chronic condition |
| `num_drug_classes` | Count of distinct drug classes prescribed |

---

## Model Architecture

Two models were trained and evaluated:

**1. Logistic Regression (baseline)**
Chosen as the primary model for interpretability. Coefficients map directly to
feature influence on non-adherence probability. Uses `class_weight="balanced"`
to account for class imbalance.

**2. Random Forest (secondary)**
Ensemble tree model providing non-linear feature interactions and feature
importance scores for cross-validation of the logistic model's findings.

---

## Performance

Evaluated on a held-out stratified test set (20% of data, n=600).

| Model | ROC-AUC | F1 (non-adherent) | Recall | Precision |
|-------|---------|-------------------|--------|-----------|
| Logistic Regression | **0.976** | 0.890 | 0.894 | 0.886 |
| Random Forest | 0.979 | 0.879 | 0.911 | 0.849 |

Non-adherence threshold: PDC < 0.80 (standard NCQA/CMS definition).

---

## Limitations

1. **Synthetic data only.** Performance on real claims data will differ. Features
   engineered from real EHR/PBM data would include richer signal (diagnosis codes,
   lab values, social determinants).

2. **No calibration.** The predicted probabilities are not calibrated. Raw scores
   should not be interpreted as precise risk percentages without Platt scaling or
   isotonic regression.

3. **Threshold sensitivity.** The 0.80 PDC threshold is a population-level
   convention. Individual patients may have legitimate reasons for gaps (hospitalization,
   drug holidays under physician guidance) that the model cannot distinguish.

4. **Temporal leakage risk in real settings.** This pipeline must ensure that
   features derived from refill history do not inadvertently use data from within
   the label window. The current implementation separates the 180-day feature window
   from the 90-day label window.

5. **No fairness evaluation.** The model has not been audited for differential
   performance across demographic subgroups. Any real deployment would require
   fairness checks across age, gender, and race/ethnicity proxies.

---

## Ethical Considerations

- This model should **assist**, not replace, clinical judgment.
- Outreach prioritization based on this model should be reviewed to avoid
  over-indexing on patients who are systematically disadvantaged (cost barriers,
  transportation, health literacy).
- False positives may waste care coordinator resources; false negatives may miss
  patients who genuinely need intervention. The operating threshold should be
  chosen based on the cost ratio of each error type in the specific deployment.

---

## Recommended Next Steps

- Calibrate predicted probabilities using cross-validated Platt scaling
- Add social determinants of health features (ZIP-level SES proxies)
- Implement subgroup fairness analysis before production deployment
- Evaluate on real de-identified claims data with proper IRB approval
- Add a time-series component to capture adherence trajectory, not just snapshot

---

*Model developed by Pavan Kishore Chaparla as a portfolio project.*
*No real patient data was used. Not for clinical use.*
