"""Create a report-ready target distribution plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.load_data import read_interim_credit_csv
from src.utils.paths import FIGURES_DIR


def main(output_name: str = "target_distribution.png") -> Path:
    """Save a clean target distribution plot."""
    df = read_interim_credit_csv()
    counts = df["Cible"].value_counts().sort_index()
    rates = counts / counts.sum()

    labels = ["Non-defaut (0)", "Defaut (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    percents = [rates.get(0, 0.0), rates.get(1, 0.0)]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=values, color="#2f6b5f")
    plt.ylabel("Nombre d'observations")
    plt.xlabel("")
    plt.title("Distribution de la variable cible")

    for i, (v, p) in enumerate(zip(values, percents)):
        plt.text(i, v + max(values) * 0.02, f"{v} ({p:.0%})", ha="center")

    out_path = FIGURES_DIR / "plots" / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


if __name__ == "__main__":
    path = main()
    print(f"Saved plot: {path}")
