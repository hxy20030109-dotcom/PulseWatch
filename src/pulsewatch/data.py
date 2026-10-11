"""Dataset loading and leakage-aware feature selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "Machine failure"
ORDER_COLUMN = "UDI"
IDENTIFIER_COLUMNS = {"UDI", "Product ID"}
FAILURE_MODE_COLUMNS = {"TWF", "HDF", "PWF", "OSF", "RNF"}
MODEL_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
REQUIRED_COLUMNS = {
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    TARGET,
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load AI4I CSV and validate columns, labels, and identifiers."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset contains no rows")
    if frame[TARGET].isna().any() or not set(frame[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"{TARGET!r} must contain only non-null binary labels 0 and 1")
    if frame[ORDER_COLUMN].isna().any() or frame[ORDER_COLUMN].duplicated().any():
        raise ValueError(f"{ORDER_COLUMN!r} must be unique and non-null")
    return frame.sort_values(ORDER_COLUMN).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return predictive inputs, excluding identifiers and target-derived labels."""
    excluded = IDENTIFIER_COLUMNS | FAILURE_MODE_COLUMNS | {TARGET}
    return [
        column
        for column in MODEL_FEATURES
        if column in frame.columns and column not in excluded
    ]


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split sorted rows into train, validation, and test partitions."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1")
    train_end = int(len(frame) * train_fraction)
    validation_end = int(len(frame) * (train_fraction + validation_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= len(frame):
        raise ValueError("Dataset is too small for the requested split fractions")
    return frame.iloc[:train_end], frame.iloc[train_end:validation_end], frame.iloc[validation_end:]
