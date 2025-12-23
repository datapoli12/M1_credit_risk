"""XGBoost model training."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.utils.paths import ARTIFACTS_DIR


def train_xgboost(
    df: pd.DataFrame, target_col: str = "Cible"
) -> XGBClassifier:
    """Train an XGBoost classifier on processed data."""
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X, y)
    return model


def save_model(model: XGBClassifier, filename: str = "xgb_model.joblib") -> Path:
    """Save trained model to artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / filename
    joblib.dump(model, path)
    return path


def load_model(filename: str = "xgb_model.joblib") -> XGBClassifier:
    """Load trained model from artifacts."""
    path = ARTIFACTS_DIR / filename
    return joblib.load(path)
