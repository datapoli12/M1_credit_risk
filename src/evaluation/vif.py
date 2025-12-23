"""Compute VIF for processed training features."""

from __future__ import annotations

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from src.data.load_data import read_processed_train_csv
from src.utils.paths import FIGURES_DIR


def main() -> None:
    """Compute and save VIF table."""
    df = read_processed_train_csv()
    X = df.drop(columns=["Cible"]).astype(float)
    X = add_constant(X, has_constant="add")

    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif = variance_inflation_factor(X.values, i)
        rows.append({"feature": col, "vif": float(vif)})

    out = pd.DataFrame(rows).sort_values("vif", ascending=False)
    out_path = FIGURES_DIR / "tables" / "vif.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved VIF table: {out_path}")


if __name__ == "__main__":
    main()
