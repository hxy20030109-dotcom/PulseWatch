"""Join prediction events with labels and write post-prediction quality metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pulsewatch.data import load_dataset
from pulsewatch.quality import evaluate_delayed_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/prediction_log.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/raw/ai4i2020.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/delayed_quality_report.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_json(args.predictions, lines=True)
    labels = load_dataset(args.labels)
    report = evaluate_delayed_labels(predictions, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved delayed-label report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
