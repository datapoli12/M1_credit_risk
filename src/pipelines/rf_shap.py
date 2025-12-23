"""Run SHAP interpretability for Random Forest."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:  # Allow direct execution: python src/pipelines/rf_shap.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.rf_shap import main


if __name__ == "__main__":
    main()
