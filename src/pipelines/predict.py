"""Predict default for new clients using a saved model and preprocessor."""

from __future__ import annotations

import sys
from pathlib import Path

import argparse
import pandas as pd

if __package__ is None:  # Allow direct execution: python src/pipelines/predict.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.preprocess import load_preprocessor
from src.models.logit import load_model, predict_proba
from src.utils.paths import PROJECT_ROOT


def main(input_path: str, output_path: str, threshold: float) -> None:
    """Generate predictions for new clients."""
    df = pd.read_csv(PROJECT_ROOT / input_path)
    preprocessor = load_preprocessor()
    processed = preprocessor.transform(df, require_target=False)

    model = load_model()
    scores = predict_proba(model, processed)
    preds = (scores >= threshold).astype(int)

    out = df.copy()
    out["pd_hat"] = scores
    out["default_pred"] = preds

    out_path = PROJECT_ROOT / output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score new clients with logit.")
    parser.add_argument(
        "--input_path",
        default="data/interim/credit_recode.csv",
        help="Path to input CSV without target column.",
    )
    parser.add_argument(
        "--output_path",
        default="data/processed/predictions.csv",
        help="Path to output CSV with predictions.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold applied to pd_hat.",
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.threshold)
