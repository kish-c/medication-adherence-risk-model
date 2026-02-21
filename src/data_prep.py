"""
Cleans the feature file and creates stratified train / test splits.

Run:
    python -m src.data_prep
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURES_FILE, TRAIN_FILE, TEST_FILE,
    DATA_PROCESSED_DIR, FEATURE_COLUMNS, TARGET_COLUMN,
    TEST_SIZE, RANDOM_SEED,
)
from src.utils import get_logger, ensure_dirs

log = get_logger(__name__)


def main() -> None:
    ensure_dirs(DATA_PROCESSED_DIR)

    log.info("Loading features from %s …", FEATURES_FILE)
    df = pd.read_csv(FEATURES_FILE)

    # Drop non-feature columns
    keep = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df[keep].copy()

    # Sanity checks
    missing = df[FEATURE_COLUMNS].isnull().sum()
    if missing.any():
        log.warning("Filling %d missing values …", missing.sum())
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)

    log.info("Class distribution:\n%s", df[TARGET_COLUMN].value_counts().to_string())

    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df[TARGET_COLUMN],
    )

    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE,  index=False)

    log.info("Train → %s  (%d rows)", TRAIN_FILE, len(train))
    log.info("Test  → %s  (%d rows)", TEST_FILE,  len(test))


if __name__ == "__main__":
    main()
