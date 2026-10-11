"""Join prediction events with later labels for post-prediction quality checks."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from pulsewatch.data import ORDER_COLUMN, TARGET
from pulsewatch.metrics import classification_report

PREDICTION_COLUMNS = {ORDER_COLUMN, "failure_risk", "threshold", "model_version"}


def evaluate_delayed_labels(predictions: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Evaluate only prediction events whose labels have arrived, keyed by UDI."""
    missing = PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction log is missing columns: {sorted(missing)}")
    if ORDER_COLUMN not in labels or TARGET not in labels:
        raise ValueError(f"Labels must contain {ORDER_COLUMN!r} and {TARGET!r}")
    if predictions.empty:
        raise ValueError("Prediction log contains no records")
    if predictions[ORDER_COLUMN].isna().any() or predictions[ORDER_COLUMN].duplicated().any():
        raise ValueError("Prediction UDI values must be unique and non-null")
    if labels[ORDER_COLUMN].isna().any() or labels[ORDER_COLUMN].duplicated().any():
        raise ValueError("Label UDI values must be unique and non-null")

    scores = pd.to_numeric(predictions["failure_risk"], errors="coerce").to_numpy(float)
    thresholds = pd.to_numeric(predictions["threshold"], errors="coerce").to_numpy(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("failure_risk values must be finite probabilities in [0, 1]")
    if not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("threshold values must be finite and in [0, 1]")
    if predictions["model_version"].nunique(dropna=False) != 1:
        raise ValueError("Evaluate one model version per delayed-label report")
    if len(np.unique(thresholds)) != 1:
        raise ValueError("Evaluate one fixed alert threshold per delayed-label report")

    joined = predictions[[ORDER_COLUMN]].copy()
    joined["failure_risk"] = scores
    joined["threshold"] = thresholds
    joined = joined.merge(
        labels[[ORDER_COLUMN, TARGET]],
        on=ORDER_COLUMN,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    labeled = joined.loc[joined["_merge"] == "both"]
    y_true = labeled[TARGET].to_numpy(dtype=int)
    y_score = labeled["failure_risk"].to_numpy(dtype=float)
    threshold = float(thresholds[0])
    has_both_classes = len(np.unique(y_true)) == 2
    metrics = classification_report(y_true, y_score, threshold) if has_both_classes else None
    pending = int((joined["_merge"] == "left_only").sum())

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "complete" if pending == 0 and has_both_classes else "partial",
        "model_version": str(predictions["model_version"].iloc[0]),
        "threshold": threshold,
        "prediction_count": int(len(predictions)),
        "labeled_count": int(len(labeled)),
        "pending_label_count": pending,
        "metrics": metrics,
        "note": (
            "Metrics cover only prediction events joined to labels by UDI. "
            "This local AI4I replay demonstrates delayed-label evaluation; it is not "
            "live production telemetry."
        ),
    }
