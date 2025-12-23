"""Run SHAP interpretability for LightGBM."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:  # Allow direct execution: python src/pipelines/lgbm_shap.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.lgbm_shap import main


if __name__ == "__main__":
    main()
