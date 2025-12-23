"""SHAP interpretability for LightGBM model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from src.data.load_data import read_processed_test_csv
from src.models.lightgbm_model import load_model
from src.utils.paths import FIGURES_DIR


def _get_shap_values(model, X: pd.DataFrame):
    """Return SHAP values for the positive class."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        return shap_values[1]
    return shap_values


def main() -> None:
    """Train LightGBM and compute SHAP explanations."""
    df = read_processed_test_csv()
    model = load_model()
    X_test = df.drop(columns=["Cible"])

    shap_values = _get_shap_values(model, X_test)

    # Summary plot
    plot_path = FIGURES_DIR / "plots" / "lgbm_shap_summary.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Mean absolute SHAP importance
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_abs_shap": abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    table_path = FIGURES_DIR / "tables" / "lgbm_shap_importance.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(table_path, index=False)

    print(f"Saved SHAP plot: {plot_path}")
    print(f"Saved SHAP table: {table_path}")


if __name__ == "__main__":
    main()
