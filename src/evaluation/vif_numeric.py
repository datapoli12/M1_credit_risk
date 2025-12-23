"""Compute VIF for numeric and ordinal features only."""

from __future__ import annotations

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from src.data.load_data import read_interim_credit_csv
from src.data.preprocess import NUMERIC_COLS, ORDINAL_MAPPINGS, apply_ordinal_mappings
from src.utils.paths import FIGURES_DIR


def main() -> None:
    """Compute and save VIF table for numeric/ordinal features."""
    df = read_interim_credit_csv()
    df = apply_ordinal_mappings(df)

    cols = NUMERIC_COLS + list(ORDINAL_MAPPINGS.keys())
    X = df[cols].astype(float)
    X = X.fillna(X.median())
    X = add_constant(X, has_constant="add")

    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif = variance_inflation_factor(X.values, i)
        rows.append({"feature": col, "vif": float(vif)})

    out = pd.DataFrame(rows).sort_values("vif", ascending=False)
    out_path = FIGURES_DIR / "tables" / "vif_numeric.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved numeric VIF table: {out_path}")


if __name__ == "__main__":
    main()
