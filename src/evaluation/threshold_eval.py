"""Threshold-based evaluation for logit model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from src.data.load_data import read_processed_test_csv, read_processed_train_csv
from src.models.logit import load_model, predict_proba
from src.utils.paths import FIGURES_DIR


def _compute_metrics(y_true, y_score, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    tpr = recall
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tpr": float(tpr),
        "fpr": float(fpr),
    }


def main() -> None:
    """Evaluate logit at fixed and train-optimized thresholds."""
    train_df = read_processed_train_csv()
    test_df = read_processed_test_csv()
    model = load_model()

    y_train = train_df["Cible"]
    y_train_score = predict_proba(model, train_df)
    y_true = test_df["Cible"]
    y_score = predict_proba(model, test_df)

    fpr, tpr, thresholds = roc_curve(y_train, y_train_score)
    ks_idx = (tpr - fpr).argmax()
    ks_threshold = float(thresholds[ks_idx])

    rows = [
        _compute_metrics(y_true, y_score, threshold=0.5),
        _compute_metrics(y_true, y_score, threshold=ks_threshold),
    ]
    rows[0]["label"] = "fixed_0.5"
    rows[0]["threshold_source"] = "fixed"
    rows[1]["label"] = "ks_optimal"
    rows[1]["threshold_source"] = "train"

    metrics = pd.DataFrame(rows)
    out_path = FIGURES_DIR / "tables" / "logit_threshold_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_path, index=False)

    print(f"Saved threshold metrics: {out_path}")


if __name__ == "__main__":
    main()
