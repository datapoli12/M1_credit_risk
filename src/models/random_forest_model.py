"""Random Forest model training."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils.paths import ARTIFACTS_DIR


def train_random_forest(
    df: pd.DataFrame, target_col: str = "Cible"
) -> RandomForestClassifier:
    """Train a Random Forest classifier on processed data."""
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X, y)
    return model


def save_model(
    model: RandomForestClassifier, filename: str = "rf_model.joblib"
) -> Path:
    """Save trained model to artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / filename
    joblib.dump(model, path)
    return path


def load_model(filename: str = "rf_model.joblib") -> RandomForestClassifier:
    """Load trained model from artifacts."""
    path = ARTIFACTS_DIR / filename
    return joblib.load(path)
