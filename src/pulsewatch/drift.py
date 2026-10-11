"""Small, dependency-light input-distribution drift summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(
    reference: pd.Series, current: pd.Series, bins: int = 10, epsilon: float = 1e-6
) -> float:
    """Compute PSI using reference quantile bins for a numeric feature."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    reference_values = reference.dropna().to_numpy(dtype=float)
    current_values = current.dropna().to_numpy(dtype=float)
    if not reference_values.size or not current_values.size:
        raise ValueError("Both reference and current samples need non-null values")

    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0 if np.array_equal(reference_values, current_values) else 1_000_000.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts = np.histogram(reference_values, bins=edges)[0].astype(float)
    current_counts = np.histogram(current_values, bins=edges)[0].astype(float)
    reference_share = np.clip(
        reference_counts / reference_counts.sum(), epsilon, None
    )
    current_share = np.clip(current_counts / current_counts.sum(), epsilon, None)
    contribution = (current_share - reference_share) * np.log(current_share / reference_share)
    return float(np.sum(contribution))


def categorical_total_variation(reference: pd.Series, current: pd.Series) -> float:
    """Return half the L1 distance between categorical frequency distributions."""
    reference = reference.dropna().astype(str)
    current = current.dropna().astype(str)
    if reference.empty or current.empty:
        raise ValueError("Both reference and current samples need non-null values")
    categories = set(reference.unique()) | set(current.unique())
    reference_share = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
    current_share = current.value_counts(normalize=True).reindex(categories, fill_value=0)
    return float(0.5 * (reference_share - current_share).abs().sum())
