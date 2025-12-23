"""Plot heatmap of Cramer's V matrix for categorical variables."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.paths import FIGURES_DIR


def main(output_name: str = "cramers_v_heatmap.png") -> None:
    """Save a Cramer's V heatmap."""
    matrix_path = FIGURES_DIR / "tables" / "cramers_v_matrix.csv"
    matrix = pd.read_csv(matrix_path, index_col=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="YlGnBu", vmin=0, vmax=1, linewidths=0.2)
    plt.title("Cramer's V - Associations entre variables categorielles")
    plt.tight_layout()

    out_path = FIGURES_DIR / "plots" / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved heatmap: {out_path}")


if __name__ == "__main__":
    main()
