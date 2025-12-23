"""Run cross-validated evaluation for logit."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:  # Allow direct execution: python src/pipelines/evaluate_logit_cv.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.logit_cv import main


if __name__ == "__main__":
    main()
