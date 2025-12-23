"""Confusion matrix evaluation for logit at chosen thresholds."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

from src.data.load_data import read_processed_test_csv, read_processed_train_csv
from src.models.logit import load_model, predict_proba
from src.utils.paths import FIGURES_DIR


def _compute_confusion(y_true, y_score, threshold: float) -> tuple[pd.DataFrame, dict[str, float]]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tp + tn + fp + fn
    metrics = {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "accuracy": float((tp + tn) / total) if total else 0.0,
    }
    matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["True 0", "True 1"],
        columns=["Pred 0", "Pred 1"],
    )
    return matrix, metrics


def _save_heatmap(matrix: pd.DataFrame, path: Path, title: str) -> None:
    plt.figure(figsize=(4, 3))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    """Compute confusion matrices for fixed and KS-optimal thresholds."""
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

    rows = []
    for label, thr in [("fixed_0.5", 0.5), ("ks_optimal", ks_threshold)]:
        matrix, metrics = _compute_confusion(y_true, y_score, thr)
        metrics["label"] = label
        rows.append(metrics)

        plot_path = FIGURES_DIR / "plots" / f"logit_confusion_{label}.png"
        _save_heatmap(matrix, plot_path, f"Logit - Confusion ({label})")

        table_path = FIGURES_DIR / "tables" / f"logit_confusion_{label}.csv"
        matrix.to_csv(table_path)

    summary = pd.DataFrame(rows)
    summary_path = FIGURES_DIR / "tables" / "logit_confusion_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved confusion summary: {summary_path}")


if __name__ == "__main__":
    main()
