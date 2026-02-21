"""
Trains Logistic Regression (baseline) and Random Forest models,
then persists them to disk.

Run:
    python -m src.train
"""

import os
import pickle

import pandas as pd
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline

from src.config import (
    TRAIN_FILE, DATA_PROCESSED_DIR,
    FEATURE_COLUMNS, TARGET_COLUMN,
    LOGISTIC_PARAMS, RANDOM_FOREST_PARAMS,
)
from src.utils import get_logger, ensure_dirs

log = get_logger(__name__)

MODELS_DIR = os.path.join(DATA_PROCESSED_DIR, "models")


def _save(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log.info("  Saved → %s", path)


def main() -> None:
    ensure_dirs(MODELS_DIR)

    log.info("Loading training data …")
    train = pd.read_csv(TRAIN_FILE)
    X = train[FEATURE_COLUMNS]
    y = train[TARGET_COLUMN]

    log.info("Training Logistic Regression …")
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(**LOGISTIC_PARAMS)),
    ])
    lr_pipeline.fit(X, y)
    _save(lr_pipeline, os.path.join(MODELS_DIR, "logistic_regression.pkl"))

    log.info("Training Random Forest …")
    rf_pipeline = Pipeline([
        ("clf", RandomForestClassifier(**RANDOM_FOREST_PARAMS)),
    ])
    rf_pipeline.fit(X, y)
    _save(rf_pipeline, os.path.join(MODELS_DIR, "random_forest.pkl"))

    log.info("Training complete. Both models saved to %s", MODELS_DIR)


if __name__ == "__main__":
    main()
