"""Evaluate the Random Forest model."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:  # Allow direct execution: python src/pipelines/evaluate_random_forest.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.rf_eval import main


if __name__ == "__main__":
    main()
