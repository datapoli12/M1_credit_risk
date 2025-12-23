"""Train the XGBoost model."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:  # Allow direct execution: python src/pipelines/train_xgboost.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.load_data import read_processed_train_csv
from src.models.xgboost_model import save_model, train_xgboost


def main() -> None:
    """Entry point for XGBoost training."""
    df = read_processed_train_csv()
    model = train_xgboost(df)
    model_path = save_model(model)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
