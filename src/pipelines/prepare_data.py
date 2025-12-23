"""Prepare processed dataset for modeling."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:  # Allow direct execution: python src/pipelines/prepare_data.py
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sklearn.model_selection import train_test_split

from src.config.settings import RANDOM_STATE, TEST_SIZE
from src.data.load_data import read_interim_credit_csv
from src.data.preprocess import Preprocessor, save_preprocessor, save_processed


def main() -> None:
    """Entry point for data preparation pipeline."""
    df = read_interim_credit_csv()
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["Cible"]
    )
    preprocessor = Preprocessor()
    preprocessor.fit(train_df)
    train_processed = preprocessor.transform(train_df)
    test_processed = preprocessor.transform(test_df)

    train_path = save_processed(train_processed, filename="credit_train_processed.csv")
    test_path = save_processed(test_processed, filename="credit_test_processed.csv")
    prep_path = save_preprocessor(preprocessor)

    print(f"Saved processed training dataset: {train_path}")
    print(f"Saved processed test dataset: {test_path}")
    print(f"Saved preprocessor: {prep_path}")


if __name__ == "__main__":
    main()
