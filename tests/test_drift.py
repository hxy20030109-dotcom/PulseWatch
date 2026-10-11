import numpy as np
import pandas as pd
import pytest

from pulsewatch.drift import categorical_total_variation, population_stability_index


def test_psi_is_zero_for_identical_numeric_distributions() -> None:
    values = pd.Series(np.arange(100, dtype=float))

    assert population_stability_index(values, values) == pytest.approx(0.0)


def test_psi_increases_for_shifted_numeric_distribution() -> None:
    reference = pd.Series(np.linspace(0.0, 1.0, 100))
    shifted = pd.Series(np.linspace(2.0, 3.0, 100))

    assert population_stability_index(reference, shifted) > 0.2


def test_total_variation_is_zero_for_identical_categories() -> None:
    values = pd.Series(["L", "M", "M", "H"])

    assert categorical_total_variation(values, values) == pytest.approx(0.0)


def test_total_variation_is_bounded_and_detects_shift() -> None:
    reference = pd.Series(["L", "M", "H", "M"])
    shifted = pd.Series(["H", "H", "H", "H"])
    score = categorical_total_variation(reference, shifted)

    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.75)


def test_psi_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="non-null"):
        population_stability_index(pd.Series([np.nan]), pd.Series([1.0]))
