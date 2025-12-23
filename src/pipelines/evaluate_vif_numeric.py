"""Run VIF diagnostics for numeric/ordinal features."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:  # Allow direct execution: python src/pipelines/evaluate_vif_numeric.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.vif_numeric import main


if __name__ == "__main__":
    main()
