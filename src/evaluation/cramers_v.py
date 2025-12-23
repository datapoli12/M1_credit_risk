"""Compute Cramer's V for nominal categorical variables."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.config.settings import MISSING_NOMINAL_LABEL
from src.data.load_data import read_interim_credit_csv
from src.data.preprocess import NOMINAL_COLS
from src.utils.paths import FIGURES_DIR


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = table.shape
    denom = max(min(k - 1, r - 1), 1)
    return float(np.sqrt(phi2 / denom))


def main() -> None:
    """Compute and save Cramer's V matrix and long table."""
    df = read_interim_credit_csv()
    df = df.copy()
    df[NOMINAL_COLS] = df[NOMINAL_COLS].fillna(MISSING_NOMINAL_LABEL)

    pairs = []
    for a, b in itertools.combinations(NOMINAL_COLS, 2):
        val = _cramers_v(df[a], df[b])
        pairs.append({"var_1": a, "var_2": b, "cramers_v": val})

    long_df = pd.DataFrame(pairs).sort_values("cramers_v", ascending=False)
    long_path = FIGURES_DIR / "tables" / "cramers_v_long.csv"
    long_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(long_path, index=False)

    # Matrix form
    matrix = pd.DataFrame(1.0, index=NOMINAL_COLS, columns=NOMINAL_COLS)
    for row in pairs:
        matrix.loc[row["var_1"], row["var_2"]] = row["cramers_v"]
        matrix.loc[row["var_2"], row["var_1"]] = row["cramers_v"]

    matrix_path = FIGURES_DIR / "tables" / "cramers_v_matrix.csv"
    matrix.to_csv(matrix_path, index=True)

    print(f"Saved Cramer's V (long): {long_path}")
    print(f"Saved Cramer's V (matrix): {matrix_path}")


if __name__ == "__main__":
    main()
