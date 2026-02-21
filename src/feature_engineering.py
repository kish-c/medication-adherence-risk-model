"""
Feature engineering: computes PDC, MPR, refill gaps, and other adherence
metrics from the raw refill / claims data.

This mirrors the logic in sql/feature_engineering.sql but runs in Python
so the full pipeline needs no database.

Run:
    python -m src.feature_engineering
"""

import numpy as np
import pandas as pd

from src.config import (
    PATIENTS_FILE, REFILLS_FILE,
    FEATURES_FILE, DATA_PROCESSED_DIR,
    OBSERVATION_START, OBSERVATION_END,
    TARGET_COLUMN,
)
from src.utils import get_logger, ensure_dirs

log = get_logger(__name__)

OBS_START = pd.Timestamp(OBSERVATION_START)
OBS_END   = pd.Timestamp(OBSERVATION_END)
PDC_WINDOW_DAYS = 180  # look-back window for PDC/MPR calculation


def _compute_pdc(refills: pd.DataFrame, window_days: int = PDC_WINDOW_DAYS) -> pd.Series:
    """
    Proportion of Days Covered (PDC) per patient.

    PDC = (unique days covered by at least one fill) / window_days
    Each fill covers [fill_date, fill_date + days_supply).
    Overlapping fills don't double-count (capped at 1 per day).
    """
    results = {}
    window_end = OBS_END
    window_start = window_end - pd.Timedelta(days=window_days)

    for pid, grp in refills.groupby("patient_id"):
        covered = np.zeros(window_days, dtype=np.int8)
        for _, row in grp.iterrows():
            fill = pd.Timestamp(row["fill_date"])
            supply = int(row["days_supply"])
            start_offset = max(0, (fill - window_start).days)
            end_offset   = min(window_days, (fill - window_start).days + supply)
            if end_offset > start_offset:
                covered[start_offset:end_offset] = 1
        results[pid] = covered.sum() / window_days

    return pd.Series(results, name="prior_pdc")


def _compute_mpr(refills: pd.DataFrame, window_days: int = PDC_WINDOW_DAYS) -> pd.Series:
    """
    Medication Possession Ratio (MPR) per patient.

    MPR = total days supply dispensed / window_days
    Can exceed 1.0 if patient stockpiles; we cap at 1.0.
    """
    window_end   = OBS_END
    window_start = window_end - pd.Timedelta(days=window_days)

    # Only count fills within the window
    mask = (pd.to_datetime(refills["fill_date"]) >= window_start) & \
           (pd.to_datetime(refills["fill_date"]) <= window_end)
    in_window = refills[mask]

    total_supply = in_window.groupby("patient_id")["days_supply"].sum()
    mpr = (total_supply / window_days).clip(upper=1.0)
    mpr.name = "prior_mpr"
    return mpr


def _compute_gap_features(refills: pd.DataFrame) -> pd.DataFrame:
    """Per-patient refill gap statistics."""
    agg = refills.groupby("patient_id").agg(
        avg_refill_gap_days=("refill_gap_days", "mean"),
        refill_gap_std=("refill_gap_days", "std"),
        max_refill_gap_days=("refill_gap_days", "max"),
        overlap_days=("overlap_days", "sum"),
        num_fills=("refill_id", "count"),
    ).reset_index()
    agg["refill_gap_std"] = agg["refill_gap_std"].fillna(0)
    return agg


def _compute_rx_features(refills: pd.DataFrame) -> pd.DataFrame:
    """Prescription-level features per patient."""
    rx_agg = refills.groupby("patient_id").agg(
        num_prescriptions=("rx_id", pd.Series.nunique),
        avg_days_supply=("days_supply", "mean"),
        num_drug_classes=("drug_class", pd.Series.nunique),
        switch_count=("drug_class", lambda x: x.nunique() - 1),
    ).reset_index()
    return rx_agg


def _build_label(refills: pd.DataFrame, patients: pd.DataFrame) -> pd.Series:
    """
    Label: non_adherent_90d

    A patient is flagged non-adherent if their PDC in the final 90-day
    observation window falls below 0.80 — the standard clinical threshold.
    """
    window_days = 90
    window_end   = OBS_END
    window_start = window_end - pd.Timedelta(days=window_days)

    covered_days = {}
    for pid, grp in refills.groupby("patient_id"):
        covered = np.zeros(window_days, dtype=np.int8)
        for _, row in grp.iterrows():
            fill   = pd.Timestamp(row["fill_date"])
            supply = int(row["days_supply"])
            start_offset = max(0, (fill - window_start).days)
            end_offset   = min(window_days, (fill - window_start).days + supply)
            if end_offset > start_offset:
                covered[start_offset:end_offset] = 1
        covered_days[pid] = covered.sum()

    pdc_90 = pd.Series(covered_days) / window_days

    # Patients with no refills in the final window → non-adherent
    all_pids = patients["patient_id"]
    pdc_90 = pdc_90.reindex(all_pids, fill_value=0.0)

    label = (pdc_90 < 0.80).astype(int)
    label.name = TARGET_COLUMN
    return label


def main() -> None:
    ensure_dirs(DATA_PROCESSED_DIR)

    log.info("Loading raw data …")
    patients = pd.read_csv(PATIENTS_FILE)
    refills  = pd.read_csv(REFILLS_FILE)

    log.info("Computing PDC …")
    pdc = _compute_pdc(refills)

    log.info("Computing MPR …")
    mpr = _compute_mpr(refills)

    log.info("Computing gap features …")
    gap_features = _compute_gap_features(refills)

    log.info("Computing prescription features …")
    rx_features = _compute_rx_features(refills)

    log.info("Building label …")
    label = _build_label(refills, patients)

    # Merge everything onto the patient base
    features = (
        patients
        .merge(gap_features, on="patient_id", how="left")
        .merge(rx_features,  on="patient_id", how="left")
        .set_index("patient_id")
        .join(pdc)
        .join(mpr)
        .join(label)
        .reset_index()
    )

    # Fill patients who had zero activity
    numeric_cols = features.select_dtypes(include="number").columns
    features[numeric_cols] = features[numeric_cols].fillna(0)

    features.to_csv(FEATURES_FILE, index=False)
    log.info("Feature file → %s  (%d rows, %d cols)",
             FEATURES_FILE, len(features), features.shape[1])

    adherence_rate = 1 - features[TARGET_COLUMN].mean()
    log.info("Adherence rate in dataset: %.1f%%", adherence_rate * 100)


if __name__ == "__main__":
    main()
