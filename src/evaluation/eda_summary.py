"""Produce EDA summary tables for reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.paths import FIGURES_DIR, INTERIM_DIR


def load_interim(path: Path = INTERIM_DIR / "credit_recode.csv") -> pd.DataFrame:
    """Load the interim dataset."""
    return pd.read_csv(path)


def summarize_target(df: pd.DataFrame, target: str = "Cible") -> pd.DataFrame:
    """Return target counts and rates."""
    counts = df[target].value_counts().sort_index()
    rates = df[target].value_counts(normalize=True).sort_index()
    summary = pd.DataFrame({target: counts.index, "count": counts.values, "rate": rates.values})
    return summary


def summarize_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Return numeric descriptive statistics."""
    return df[numeric_cols].describe().T.reset_index().rename(columns={"index": "variable"})


def summarize_categorical(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """Return categorical counts in long format."""
    rows = []
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False)
        for category, count in vc.items():
            rows.append({"variable": col, "category": category, "count": count})
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save a table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    """Generate summary tables into figures/tables."""
    df = load_interim()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    target_summary = summarize_target(df)
    save_table(target_summary, FIGURES_DIR / "tables" / "target_summary.csv")

    numeric_summary = summarize_numeric(df, numeric_cols)
    save_table(numeric_summary, FIGURES_DIR / "tables" / "numeric_summary.csv")

    categorical_summary = summarize_categorical(df, categorical_cols)
    save_table(categorical_summary, FIGURES_DIR / "tables" / "categorical_summary.csv")


if __name__ == "__main__":
    main()
