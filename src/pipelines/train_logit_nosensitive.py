"""Train logit model without sensitive variables."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:  # Allow direct execution: python src/pipelines/train_logit_nosensitive.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.load_data import read_processed_train_csv
from src.models.logit import save_model, save_summary, train_logit


SENSITIVE_PREFIXES = ("Situation_familiale_", "Etranger_")


def _drop_sensitive(df):
    cols = [c for c in df.columns if c.startswith(SENSITIVE_PREFIXES)]
    return df.drop(columns=cols)


def main() -> None:
    """Entry point for no-sensitive logit training."""
    df = read_processed_train_csv()
    df = _drop_sensitive(df)
    model = train_logit(df)
    model_path = save_model(model, filename="logit_model_nosensitive.joblib")
    summary_path = save_summary(model, filename="logit_summary_nosensitive.txt")
    print(f"Saved model: {model_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
