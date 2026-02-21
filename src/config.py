"""
Central configuration for the medication adherence risk model pipeline.
Adjust paths, random seeds, and model parameters here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR        = os.path.join(BASE_DIR, "reports")
FIGURES_DIR        = os.path.join(REPORTS_DIR, "figures")

PATIENTS_FILE      = os.path.join(DATA_RAW_DIR, "patients.csv")
PRESCRIPTIONS_FILE = os.path.join(DATA_RAW_DIR, "prescriptions.csv")
REFILLS_FILE       = os.path.join(DATA_RAW_DIR, "refills.csv")

FEATURES_FILE      = os.path.join(DATA_PROCESSED_DIR, "features.csv")
TRAIN_FILE         = os.path.join(DATA_PROCESSED_DIR, "train.csv")
TEST_FILE          = os.path.join(DATA_PROCESSED_DIR, "test.csv")

# ── Data generation ────────────────────────────────────────────────────────
RANDOM_SEED       = 42
N_PATIENTS        = 3000
OBSERVATION_START = "2022-01-01"
OBSERVATION_END   = "2023-12-31"

DRUG_CLASSES = [
    "antihypertensive",
    "antidiabetic",
    "statin",
    "antidepressant",
    "anticoagulant",
]

DAYS_SUPPLY_OPTIONS = [30, 60, 90]

# ── Model training ─────────────────────────────────────────────────────────
TEST_SIZE         = 0.2
TARGET_COLUMN     = "non_adherent_90d"

FEATURE_COLUMNS = [
    "age",
    "chronic_flag",
    "num_prescriptions",
    "avg_days_supply",
    "prior_mpr",
    "prior_pdc",
    "avg_refill_gap_days",
    "refill_gap_std",
    "switch_count",
    "overlap_days",
    "num_drug_classes",
    "max_refill_gap_days",
]

LOGISTIC_PARAMS = {
    "C": 0.5,
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_leaf": 10,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
    "n_jobs": -1,
}

# ── Reporting ──────────────────────────────────────────────────────────────
FIGURE_DPI    = 150
FIGURE_STYLE  = "seaborn-v0_8-whitegrid"
PALETTE       = {"non-adherent": "#E05C5C", "adherent": "#5B8DB8"}
