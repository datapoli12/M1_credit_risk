"""Prepare interim dataset from raw data."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:  # Allow direct execution: python src/pipelines/train.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.load_data import read_raw_credit_csv
from src.data.preprocess import recode_target, save_interim


def main() -> None:
    """Entry point for interim data preparation."""
    df = read_raw_credit_csv()
    df = recode_target(df)
    path = save_interim(df)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Saved interim dataset: {path}")


if __name__ == "__main__":
    main()
