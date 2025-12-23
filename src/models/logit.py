"""Logit reference model training."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import statsmodels.api as sm

from src.utils.paths import ARTIFACTS_DIR, REPORTS_DIR


def prepare_design_matrix(
    df: pd.DataFrame, target_col: str = "Cible", require_target: bool = True
) -> tuple[pd.Series | None, pd.DataFrame]:
    """Prepare y and X with a constant and numeric types."""
    if target_col not in df.columns:
        if require_target:
            raise KeyError(f"Target column not found: {target_col}")
        y = None
        X = df.astype(float)
    else:
        y = df[target_col]
        X = df.drop(columns=[target_col]).astype(float)
    X = sm.add_constant(X, has_constant="add")
    return y, X


def train_logit(
    df: pd.DataFrame, target_col: str = "Cible"
) -> sm.discrete.discrete_model.BinaryResultsWrapper:
    """Train a reference logit model on processed data."""
    y, X = prepare_design_matrix(df, target_col=target_col)
    model = sm.Logit(y, X)
    return model.fit(disp=0)


def predict_proba(
    model: sm.discrete.discrete_model.BinaryResultsWrapper,
    df: pd.DataFrame,
    target_col: str = "Cible",
) -> pd.Series:
    """Predict default probabilities using a fitted logit model."""
    _, X = prepare_design_matrix(df, target_col=target_col, require_target=False)
    X = X.reindex(columns=model.params.index, fill_value=0.0)
    return model.predict(X)


def save_model(
    model: sm.discrete.discrete_model.BinaryResultsWrapper,
    filename: str = "logit_model.joblib",
) -> Path:
    """Save trained model to artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / filename
    joblib.dump(model, path)
    return path


def load_model(
    filename: str = "logit_model.joblib",
) -> sm.discrete.discrete_model.BinaryResultsWrapper:
    """Load trained model from artifacts."""
    path = ARTIFACTS_DIR / filename
    return joblib.load(path)


def save_summary(
    model: sm.discrete.discrete_model.BinaryResultsWrapper,
    filename: str = "logit_summary.txt",
) -> Path:
    """Save model summary for reporting."""
    path = REPORTS_DIR / "draft" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.summary().as_text(), encoding="utf-8")
    return path
