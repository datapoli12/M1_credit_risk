"""Evaluate Random Forest performance (ROC/AUC/KS)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from src.data.load_data import read_processed_test_csv
from src.models.random_forest_model import load_model
from src.utils.paths import FIGURES_DIR


def compute_ks(fpr: pd.Series, tpr: pd.Series) -> float:
    """Compute KS statistic from ROC curve."""
    return float((tpr - fpr).max())


def save_roc_curve(y_true: pd.Series, y_score: pd.Series, path: Path) -> None:
    """Save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC - Random Forest")
    plt.legend(loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    """Run evaluation and save metrics/plots."""
    df = read_processed_test_csv()
    model = load_model()

    y_true = df["Cible"]
    X_test = df.drop(columns=["Cible"])
    y_score = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = compute_ks(pd.Series(fpr), pd.Series(tpr))

    metrics = pd.DataFrame([{"auc": auc, "ks": ks}])
    metrics_path = FIGURES_DIR / "tables" / "rf_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)

    save_roc_curve(y_true, y_score, FIGURES_DIR / "plots" / "rf_roc.png")

    # Feature importance
    importance = pd.DataFrame(
        {"feature": X_test.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_path = FIGURES_DIR / "tables" / "rf_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved ROC: {FIGURES_DIR / 'plots' / 'rf_roc.png'}")
    print(f"Saved importance: {importance_path}")


if __name__ == "__main__":
    main()
