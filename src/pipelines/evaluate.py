"""Evaluate trained models and generate metrics."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:  # Allow direct execution: python src/pipelines/evaluate.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation import (
    lgbm_eval,
    lgbm_shap,
    logit_cv,
    logit_eval,
    logit_eval_nosensitive,
    rf_eval,
    rf_shap,
    threshold_eval,
    vif,
    vif_numeric,
    cramers_v,
    confusion_matrix_eval,
    xgb_eval,
    xgb_shap,
)
from src.utils.paths import ARTIFACTS_DIR


def _require_artifacts() -> None:
    """Ensure trained models exist before evaluation."""
    required = [
        "logit_model.joblib",
        "logit_model_nosensitive.joblib",
        "xgb_model.joblib",
        "rf_model.joblib",
        "lgbm_model.joblib",
    ]
    missing = [name for name in required if not (ARTIFACTS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts. Run training first: " + ", ".join(missing)
        )


def main() -> None:
    """Run all evaluation steps in sequence."""
    _require_artifacts()
    logit_eval.main()
    logit_eval_nosensitive.main()
    logit_cv.main()
    xgb_eval.main()
    rf_eval.main()
    lgbm_eval.main()
    xgb_shap.main()
    lgbm_shap.main()
    rf_shap.main()
    threshold_eval.main()
    vif.main()
    vif_numeric.main()
    cramers_v.main()
    confusion_matrix_eval.main()


if __name__ == "__main__":
    main()
