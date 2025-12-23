"""Data loading utilities."""

from pathlib import Path

import pandas as pd

from src.utils.paths import INTERIM_DIR, PROCESSED_DIR, RAW_DIR

DEFAULT_RAW_CSV = RAW_DIR / "data_for_these.csv"
DEFAULT_INTERIM_CSV = INTERIM_DIR / "credit_recode.csv"
DEFAULT_PROCESSED_CSV = PROCESSED_DIR / "credit_processed.csv"
DEFAULT_PROCESSED_TRAIN_CSV = PROCESSED_DIR / "credit_train_processed.csv"
DEFAULT_PROCESSED_TEST_CSV = PROCESSED_DIR / "credit_test_processed.csv"


def read_raw_credit_csv(path: str | Path = DEFAULT_RAW_CSV) -> pd.DataFrame:
    """Load the raw credit CSV with the correct separator."""
    return pd.read_csv(path, sep=";", quotechar="\"")


def read_interim_credit_csv(path: str | Path = DEFAULT_INTERIM_CSV) -> pd.DataFrame:
    """Load the interim credit CSV produced by preprocessing."""
    return pd.read_csv(path)


def read_processed_credit_csv(path: str | Path = DEFAULT_PROCESSED_CSV) -> pd.DataFrame:
    """Load the processed credit CSV ready for modeling."""
    return pd.read_csv(path)


def read_processed_train_csv(
    path: str | Path = DEFAULT_PROCESSED_TRAIN_CSV,
) -> pd.DataFrame:
    """Load the processed training CSV."""
    return pd.read_csv(path)


def read_processed_test_csv(
    path: str | Path = DEFAULT_PROCESSED_TEST_CSV,
) -> pd.DataFrame:
    """Load the processed test CSV."""
    return pd.read_csv(path)
