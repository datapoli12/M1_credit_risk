"""Evaluate logit without sensitive variables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from src.data.load_data import read_processed_test_csv
from src.models.logit import load_model, predict_proba
from src.utils.paths import FIGURES_DIR


SENSITIVE_PREFIXES = ("Situation_familiale_", "Etranger_")


def _drop_sensitive(df: pd.DataFrame) -> pd.DataFrame:
    """Drop sensitive variables by prefix."""
    cols = [c for c in df.columns if c.startswith(SENSITIVE_PREFIXES)]
    return df.drop(columns=cols)


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
    plt.title("ROC - Logit (No Sensitive Vars)")
    plt.legend(loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    """Train and evaluate logit without sensitive variables."""
    df = read_processed_test_csv()
    df = _drop_sensitive(df)
    model = load_model(filename="logit_model_nosensitive.joblib")

    y_true = df["Cible"]
    y_score = predict_proba(model, df)

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = compute_ks(pd.Series(fpr), pd.Series(tpr))

    metrics = pd.DataFrame([{"auc": auc, "ks": ks}])
    metrics_path = FIGURES_DIR / "tables" / "logit_metrics_nosensitive.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)

    save_roc_curve(y_true, y_score, FIGURES_DIR / "plots" / "logit_roc_nosensitive.png")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved ROC: {FIGURES_DIR / 'plots' / 'logit_roc_nosensitive.png'}")


if __name__ == "__main__":
    main()
