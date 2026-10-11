"""Compare feature distributions in two CSV batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pulsewatch.data import MODEL_FEATURES
from pulsewatch.drift import categorical_total_variation, population_stability_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/drift_report.json"))
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--tv-threshold", type=float, default=0.2)
    return parser.parse_args()


def read_batch(path: Path) -> pd.DataFrame:
    """Read model inputs; labels are optional because they may arrive later."""
    frame = pd.read_csv(path)
    missing = set(MODEL_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Batch {path} is missing model features: {sorted(missing)}")
    return frame[MODEL_FEATURES]


def compare_batches(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float,
    tv_threshold: float,
) -> dict:
    """Score numerical and categorical feature shifts against fixed thresholds."""
    if list(reference.columns) != list(current.columns):
        raise ValueError("Reference and current batches must have identical feature columns")
    results = {}
    for column in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[column]):
            score = population_stability_index(reference[column], current[column])
            metric, threshold = "psi", psi_threshold
        else:
            score = categorical_total_variation(reference[column], current[column])
            metric, threshold = "total_variation", tv_threshold
        results[column] = {
            "metric": metric,
            "score": score,
            "threshold": threshold,
            "drift_alert": score >= threshold,
        }
    return {
        "reference_rows": len(reference),
        "current_rows": len(current),
        "features": results,
        "interpretation": (
            "Distribution shift is a monitoring signal, not proof of lower model quality. "
            "Evaluate predictive metrics when ground-truth labels become available."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.psi_threshold < 0 or args.tv_threshold < 0:
        raise SystemExit("Drift thresholds must be non-negative")
    reference, current = read_batch(args.reference), read_batch(args.current)
    report = compare_batches(reference, current, args.psi_threshold, args.tv_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved drift report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
