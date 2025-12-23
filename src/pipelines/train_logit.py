"""Train the reference logit model."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:  # Allow direct execution: python src/pipelines/train_logit.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.load_data import read_processed_train_csv
from src.models.logit import save_model, save_summary, train_logit


def main() -> None:
    """Entry point for logit training."""
    df = read_processed_train_csv()
    model = train_logit(df)
    model_path = save_model(model)
    summary_path = save_summary(model)
    print(f"Saved model: {model_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
