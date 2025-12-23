"""Minimal preprocessing for the raw credit dataset."""

from dataclasses import dataclass, field
import warnings
from pathlib import Path

import joblib
import pandas as pd

from src.config.settings import DROP_FIRST, MIN_PCT_RARE, MISSING_NOMINAL_LABEL, MISSING_ORDINAL_VALUE
from src.utils.paths import ARTIFACTS_DIR, INTERIM_DIR, PROCESSED_DIR


def recode_target(df: pd.DataFrame, column: str = "Cible") -> pd.DataFrame:
    """Recode target from 1/2 to 0/1 without mutating the input."""
    out = df.copy()
    out[column] = out[column].replace({1: 0, 2: 1})
    return out


def save_interim(df: pd.DataFrame, filename: str = "credit_recode.csv") -> Path:
    """Save an interim dataset into data/interim."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    path = INTERIM_DIR / filename
    df.to_csv(path, index=False)
    return path


ORDINAL_MAPPINGS = {
    # Clear ordered scale: savings level
    "Epargne": {"A65": 0, "A61": 1, "A62": 2, "A63": 3, "A64": 4},
    # Clear ordered scale: employment length
    "Anciennete_emploi": {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4},
}

NOMINAL_COLS = [
    "Comptes",
    "Historique_credit",
    "Objet_credit",
    "Situation_familiale",
    "Garanties",
    "Biens",
    "Autres_credits",
    "Statut_domicile",
    "Type_emploi",
    "Telephone",
    "Etranger",
]

NUMERIC_COLS = [
    "Duree_credit",
    "Montant_credit",
    "Taux_effort",
    "Anciennete_domicile",
    "Age",
    "Nb_credits",
    "Nb_pers_charge",
]


def _validate_columns(df: pd.DataFrame, cols: list[str]) -> None:
    """Ensure required columns exist."""
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")


def _validate_target(
    df: pd.DataFrame, target_col: str, require_target: bool = True
) -> None:
    """Validate target is binary and non-null when required."""
    if target_col not in df.columns:
        if require_target:
            raise KeyError(f"Target column not found: {target_col}")
        return
    if df[target_col].isna().any():
        raise ValueError(f"Target column has missing values: {target_col}")
    values = set(df[target_col].unique())
    if not values.issubset({0, 1}):
        raise ValueError(f"Target column must be binary 0/1, got: {sorted(values)}")


def _fill_missing_nominal(
    df: pd.DataFrame, cols: list[str], missing_label: str
) -> pd.DataFrame:
    """Fill missing values for nominal columns."""
    out = df.copy()
    out[cols] = out[cols].fillna(missing_label)
    return out


def _apply_ordinal_mappings(
    df: pd.DataFrame,
    missing_ordinal_value: int,
    ordinal_mappings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Apply ordinal mappings with validation and missing handling."""
    out = df.copy()
    for col, mapping in ordinal_mappings.items():
        if col not in out.columns:
            raise KeyError(f"Missing column for ordinal mapping: {col}")
        non_null = out[col].dropna().unique()
        unmapped = set(non_null) - set(mapping.keys())
        if unmapped:
            raise ValueError(f"Unmapped categories in {col}: {sorted(unmapped)}")
        out[col] = out[col].map(mapping)
        out[col] = out[col].fillna(missing_ordinal_value)
    return out


@dataclass
class Preprocessor:
    """Fit/transform preprocessing to avoid data leakage."""

    target_col: str = "Cible"
    nominal_cols: list[str] = field(default_factory=lambda: NOMINAL_COLS)
    ordinal_mappings: dict[str, dict[str, int]] = field(
        default_factory=lambda: ORDINAL_MAPPINGS
    )
    numeric_cols: list[str] = field(default_factory=lambda: NUMERIC_COLS)
    drop_first: bool = DROP_FIRST
    min_pct: float = MIN_PCT_RARE
    other_label: str = "Other"
    missing_nominal_label: str = MISSING_NOMINAL_LABEL
    missing_ordinal_value: int = MISSING_ORDINAL_VALUE

    rare_map: dict[str, set] = field(default_factory=dict)
    known_categories: dict[str, set] = field(default_factory=dict)
    numeric_impute_values: dict[str, float] = field(default_factory=dict)
    dummy_columns: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Fit preprocessing parameters on training data."""
        _validate_target(df, self.target_col, require_target=True)
        _validate_columns(
            df, self.nominal_cols + list(self.ordinal_mappings.keys()) + self.numeric_cols
        )

        working = _fill_missing_nominal(df, self.nominal_cols, self.missing_nominal_label)
        self.known_categories = {
            col: set(working[col].unique()) for col in self.nominal_cols
        }
        self.rare_map = fit_rare_categories(
            working, self.nominal_cols, min_pct=self.min_pct
        )
        self.numeric_impute_values = {
            col: float(working[col].median()) for col in self.numeric_cols
        }

        transformed = self._transform_core(working, fit=True)
        self.dummy_columns = [
            col for col in transformed.columns if col != self.target_col
        ]
        return self

    def transform(self, df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
        """Transform data using fitted preprocessing parameters."""
        if not self.dummy_columns:
            raise ValueError("Preprocessor must be fitted before transform.")
        _validate_target(df, self.target_col, require_target=require_target)
        working = _fill_missing_nominal(df, self.nominal_cols, self.missing_nominal_label)
        return self._transform_core(working, fit=False)

    def _transform_core(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Core transform shared by fit and transform."""
        out = df.copy()
        if not fit:
            for col in self.nominal_cols:
                known = self.known_categories.get(col, set())
                out[col] = out[col].where(out[col].isin(known), self.other_label)
        out = apply_rare_categories(out, self.rare_map, other_label=self.other_label)
        out = _apply_ordinal_mappings(
            out, self.missing_ordinal_value, self.ordinal_mappings
        )
        for col, value in self.numeric_impute_values.items():
            out[col] = out[col].fillna(value)
        out = pd.get_dummies(out, columns=self.nominal_cols, drop_first=self.drop_first)
        bool_cols = out.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            out[bool_cols] = out[bool_cols].astype(int)
        if self.target_col in out.columns:
            target = out[self.target_col]
            out = out.drop(columns=[self.target_col])
            out[self.target_col] = target
        if not fit:
            if self.target_col in out.columns:
                out = out.reindex(
                    columns=self.dummy_columns + [self.target_col], fill_value=0
                )
            else:
                out = out.reindex(columns=self.dummy_columns, fill_value=0)
        return out


def save_preprocessor(
    preprocessor: Preprocessor, filename: str = "credit_preprocessor.joblib"
) -> Path:
    """Save fitted preprocessor to artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / filename
    joblib.dump(preprocessor, path)
    return path


def load_preprocessor(filename: str = "credit_preprocessor.joblib") -> Preprocessor:
    """Load a fitted preprocessor from artifacts."""
    path = ARTIFACTS_DIR / filename
    return joblib.load(path)

def fit_rare_categories(
    df: pd.DataFrame, cols: list[str], min_pct: float = 0.05
) -> dict[str, set]:
    """Identify rare categories by column using a frequency threshold."""
    rare_map: dict[str, set] = {}
    n = len(df)
    for col in cols:
        freq = df[col].value_counts(dropna=False) / n
        rare_map[col] = set(freq[freq < min_pct].index)
    return rare_map


def apply_rare_categories(
    df: pd.DataFrame, rare_map: dict[str, set], other_label: str = "Other"
) -> pd.DataFrame:
    """Apply a precomputed rare-category map to group categories."""
    out = df.copy()
    for col, rare in rare_map.items():
        if col in out.columns and rare:
            out[col] = out[col].where(~out[col].isin(rare), other_label)
    return out


def group_rare_categories(
    df: pd.DataFrame,
    cols: list[str],
    min_pct: float = 0.05,
    other_label: str = "Other",
) -> pd.DataFrame:
    """Group rare categories into a single label for stability."""
    rare_map = fit_rare_categories(df, cols, min_pct=min_pct)
    return apply_rare_categories(df, rare_map, other_label=other_label)


def apply_ordinal_mappings(
    df: pd.DataFrame,
    missing_ordinal_value: int = MISSING_ORDINAL_VALUE,
) -> pd.DataFrame:
    """Apply ordinal mappings with validation and missing handling."""
    return _apply_ordinal_mappings(
        df, missing_ordinal_value=missing_ordinal_value, ordinal_mappings=ORDINAL_MAPPINGS
    )


def build_processed_dataset(
    df: pd.DataFrame,
    target_col: str = "Cible",
    drop_first: bool = DROP_FIRST,
    rare_map: dict[str, set] | None = None,
    min_pct: float = MIN_PCT_RARE,
    missing_nominal_label: str = MISSING_NOMINAL_LABEL,
    missing_ordinal_value: int = MISSING_ORDINAL_VALUE,
) -> pd.DataFrame:
    """Build a processed dataset for a single dataset only (no hold-out split).

    Warning: use Preprocessor.fit/transform on train/test for evaluation.
    """
    warnings.warn(
        "build_processed_dataset is for single-dataset use only; "
        "use Preprocessor.fit/transform for train/test evaluation.",
        UserWarning,
    )
    if rare_map is None:
        rare_map = fit_rare_categories(df, NOMINAL_COLS, min_pct=min_pct)
    out = _fill_missing_nominal(df, NOMINAL_COLS, missing_nominal_label)
    out = apply_rare_categories(out, rare_map)
    out = apply_ordinal_mappings(out, missing_ordinal_value=missing_ordinal_value)
    for col in NUMERIC_COLS:
        out[col] = out[col].fillna(float(out[col].median()))
    out = pd.get_dummies(out, columns=NOMINAL_COLS, drop_first=drop_first)
    bool_cols = out.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        out[bool_cols] = out[bool_cols].astype(int)
    if target_col in out.columns:
        target = out[target_col]
        out = out.drop(columns=[target_col])
        out[target_col] = target
    return out


def save_processed(df: pd.DataFrame, filename: str = "credit_processed.csv") -> Path:
    """Save a processed dataset into data/processed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    return path
