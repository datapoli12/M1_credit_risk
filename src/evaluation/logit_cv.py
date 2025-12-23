"""Cross-validated evaluation for logit (AUC/KS) with leakage-free preprocessing."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from src.data.load_data import read_interim_credit_csv
from src.data.preprocess import Preprocessor
from src.models.logit import predict_proba, train_logit
from src.utils.paths import FIGURES_DIR


def compute_ks(fpr: pd.Series, tpr: pd.Series) -> float:
    """Compute KS statistic from ROC curve."""
    return float((tpr - fpr).max())


def main() -> None:
    """Run stratified CV and save fold metrics."""
    df = read_interim_credit_csv()
    X = df.drop(columns=["Cible"])
    y = df["Cible"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        preprocessor = Preprocessor()
        preprocessor.fit(train_df)
        train_processed = preprocessor.transform(train_df)
        test_processed = preprocessor.transform(test_df)

        model = train_logit(train_processed)
        y_true = test_processed["Cible"]
        y_score = predict_proba(model, test_processed)
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ks = compute_ks(pd.Series(fpr), pd.Series(tpr))
        rows.append({"fold": i, "auc": auc, "ks": ks})

    metrics = pd.DataFrame(rows)
    summary = metrics[["auc", "ks"]].agg(["mean", "std"]).reset_index()
    out_path = FIGURES_DIR / "tables" / "logit_cv_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_path, index=False)

    summary_path = FIGURES_DIR / "tables" / "logit_cv_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved CV metrics: {out_path}")
    print(f"Saved CV summary: {summary_path}")


if __name__ == "__main__":
    main()
