"""
Generates synthetic pharmacy claims / refill data that mimics de-identified
patient behavior without using any real PHI.

Run:
    python -m src.data_generate
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from src.config import (
    RANDOM_SEED, N_PATIENTS, OBSERVATION_START, OBSERVATION_END,
    DRUG_CLASSES, DAYS_SUPPLY_OPTIONS,
    PATIENTS_FILE, PRESCRIPTIONS_FILE, REFILLS_FILE,
    DATA_RAW_DIR,
)
from src.utils import get_logger, ensure_dirs

log = get_logger(__name__)


def _generate_patients(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Simulate a patient population with basic demographic proxies."""
    ages = rng.integers(18, 85, size=n)
    # Older patients tend to have chronic conditions more often
    chronic_prob = np.clip(0.3 + (ages - 18) / 200, 0.3, 0.85)
    chronic_flag = rng.binomial(1, chronic_prob)

    patients = pd.DataFrame({
        "patient_id": [f"P{str(i).zfill(5)}" for i in range(n)],
        "age": ages,
        "chronic_flag": chronic_flag,
        # Gender proxy (0/1), not used as a feature — kept for completeness
        "gender_proxy": rng.integers(0, 2, size=n),
    })
    return patients


def _generate_prescriptions(
    rng: np.random.Generator,
    patients: pd.DataFrame,
    obs_start: pd.Timestamp,
    obs_end: pd.Timestamp,
) -> pd.DataFrame:
    """Assign each patient 1–4 prescriptions over the observation window."""
    rows = []
    max_days = (obs_end - obs_start).days

    for _, patient in patients.iterrows():
        pid = patient["patient_id"]
        # Chronic patients tend to have more concurrent medications
        n_rx = rng.integers(2, 5) if patient["chronic_flag"] else rng.integers(1, 3)
        drug_classes = rng.choice(DRUG_CLASSES, size=n_rx, replace=False)

        for drug_class in drug_classes:
            start_offset = rng.integers(0, max(1, max_days - 90))
            start_date = obs_start + timedelta(days=int(start_offset))
            days_supply = rng.choice(DAYS_SUPPLY_OPTIONS)

            rows.append({
                "rx_id": f"RX{len(rows):06d}",
                "patient_id": pid,
                "drug_class": drug_class,
                "rx_start_date": start_date.strftime("%Y-%m-%d"),
                "days_supply": int(days_supply),
            })

    return pd.DataFrame(rows)


def _generate_refills(
    rng: np.random.Generator,
    prescriptions: pd.DataFrame,
    patients: pd.DataFrame,
    obs_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Simulate refill behavior for each prescription.
    Non-adherent patterns: longer gaps, skipped refills, early discontinuation.
    """
    patient_lookup = patients.set_index("patient_id")[["age", "chronic_flag"]]
    rows = []

    for _, rx in prescriptions.iterrows():
        pid = rx["patient_id"]
        age = patient_lookup.loc[pid, "age"]
        chronic = patient_lookup.loc[pid, "chronic_flag"]

        # Base adherence probability — influenced by age and chronic status
        base_adherence = 0.55 + (chronic * 0.15) + (min(age, 65) / 650)
        base_adherence = float(np.clip(base_adherence, 0.35, 0.82))

        current_date = pd.Timestamp(rx["rx_start_date"])
        days_supply = int(rx["days_supply"])
        fill_number = 0
        prev_end = current_date + timedelta(days=days_supply)

        while current_date <= obs_end:
            gap_days = 0 if fill_number == 0 else int(
                rng.choice(
                    [0, rng.integers(1, 8), rng.integers(8, 30), rng.integers(30, 90)],
                    p=[base_adherence * 0.6,
                       base_adherence * 0.4,
                       (1 - base_adherence) * 0.5,
                       (1 - base_adherence) * 0.5],
                )
            )

            fill_date = prev_end + timedelta(days=gap_days)
            if fill_date > obs_end:
                break

            overlap = max(0, (prev_end - fill_date).days) if fill_number > 0 else 0

            rows.append({
                "refill_id": f"RF{len(rows):07d}",
                "rx_id": rx["rx_id"],
                "patient_id": pid,
                "drug_class": rx["drug_class"],
                "fill_date": fill_date.strftime("%Y-%m-%d"),
                "days_supply": days_supply,
                "fill_number": fill_number,
                "refill_gap_days": gap_days,
                "overlap_days": overlap,
            })

            prev_end = fill_date + timedelta(days=days_supply)
            fill_number += 1

            # Simulate dropout — less likely for chronic/older patients
            dropout_prob = 0.08 if chronic else 0.15
            if rng.random() < dropout_prob:
                break

    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs(DATA_RAW_DIR)
    rng = np.random.default_rng(RANDOM_SEED)

    obs_start = pd.Timestamp(OBSERVATION_START)
    obs_end   = pd.Timestamp(OBSERVATION_END)

    log.info("Generating %d synthetic patients …", N_PATIENTS)
    patients = _generate_patients(rng, N_PATIENTS)
    patients.to_csv(PATIENTS_FILE, index=False)
    log.info("  → %s  (%d rows)", PATIENTS_FILE, len(patients))

    log.info("Generating prescriptions …")
    prescriptions = _generate_prescriptions(rng, patients, obs_start, obs_end)
    prescriptions.to_csv(PRESCRIPTIONS_FILE, index=False)
    log.info("  → %s  (%d rows)", PRESCRIPTIONS_FILE, len(prescriptions))

    log.info("Generating refills …")
    refills = _generate_refills(rng, prescriptions, patients, obs_end)
    refills.to_csv(REFILLS_FILE, index=False)
    log.info("  → %s  (%d rows)", REFILLS_FILE, len(refills))

    log.info("Data generation complete.")


if __name__ == "__main__":
    main()
