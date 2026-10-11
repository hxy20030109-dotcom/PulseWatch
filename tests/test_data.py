import pandas as pd
import pytest

from pulsewatch.data import (
    FAILURE_MODE_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET,
    chronological_split,
    feature_columns,
    load_dataset,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UDI": [3, 1, 2, 4],
            "Product ID": ["M3", "L1", "L2", "H4"],
            "Type": ["M", "L", "L", "H"],
            "Air temperature [K]": [300.0, 298.0, 299.0, 301.0],
            "Process temperature [K]": [310.0, 308.0, 309.0, 311.0],
            "Rotational speed [rpm]": [1500, 1450, 1475, 1600],
            "Torque [Nm]": [40.0, 42.0, 41.0, 38.0],
            "Tool wear [min]": [100, 20, 60, 200],
            TARGET: [0, 1, 0, 0],
            **{column: [0, 0, 0, 0] for column in FAILURE_MODE_COLUMNS},
        }
    )


def test_load_dataset_sorts_by_identifier(tmp_path) -> None:
    path = tmp_path / "records.csv"
    sample_frame().to_csv(path, index=False)

    loaded = load_dataset(path)

    assert loaded["UDI"].tolist() == [1, 2, 3, 4]


def test_load_dataset_rejects_duplicate_identifiers(tmp_path) -> None:
    frame = sample_frame()
    frame.loc[1, "UDI"] = frame.loc[0, "UDI"]
    path = tmp_path / "records.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="UDI.*unique"):
        load_dataset(path)


def test_feature_columns_exclude_ids_and_failure_mode_labels() -> None:
    columns = set(feature_columns(sample_frame()))

    assert IDENTIFIER_COLUMNS.isdisjoint(columns)
    assert FAILURE_MODE_COLUMNS.isdisjoint(columns)
    assert TARGET not in columns
    assert "Torque [Nm]" in columns


def test_chronological_split_preserves_order_and_sizes() -> None:
    frame = sample_frame().sort_values("UDI").reset_index(drop=True)

    train, validation, test = chronological_split(frame, 0.5, 0.25)

    assert [len(part) for part in (train, validation, test)] == [2, 1, 1]
    assert train["UDI"].max() < validation["UDI"].min()
    assert validation["UDI"].max() < test["UDI"].min()


def test_chronological_split_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError, match="sum to less than 1"):
        chronological_split(sample_frame(), 0.8, 0.2)
