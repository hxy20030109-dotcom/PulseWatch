"""Send held-out AI4I rows to the running API and summarize service/drift signals."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from pulsewatch.data import chronological_split, feature_columns, load_dataset
from pulsewatch.drift import categorical_total_variation, population_stability_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/ai4i2020.csv"))
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--shift-factor",
        type=float,
        default=1.0,
        help="Multiply torque and tool wear in the simulated stream to demonstrate drift.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/stream_report.json"))
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("artifacts/prediction_log.jsonl"),
        help="Write successful predictions separately from labels for later evaluation.",
    )
    return parser.parse_args()


def api_payload(row: pd.Series) -> dict[str, str | float | int]:
    """Map dataset fields to the public inference request schema."""
    return {
        "type": str(row["Type"]),
        "air_temperature_k": float(row["Air temperature [K]"]),
        "process_temperature_k": float(row["Process temperature [K]"]),
        "rotational_speed_rpm": int(row["Rotational speed [rpm]"]),
        "torque_nm": float(row["Torque [Nm]"]),
        "tool_wear_min": int(row["Tool wear [min]"]),
    }


def call_api(url: str, payload: dict) -> tuple[dict, float]:
    """Post one prediction and return the response and client-observed latency."""
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=10) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body, (time.perf_counter() - started) * 1000


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    report = {}
    for column in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[column]):
            score = population_stability_index(reference[column], current[column])
            report[column] = {"metric": "psi", "score": score, "alert": score >= 0.2}
        else:
            score = categorical_total_variation(reference[column], current[column])
            report[column] = {
                "metric": "total_variation",
                "score": score,
                "alert": score >= 0.2,
            }
    return report


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.shift_factor <= 0:
        raise SystemExit("--shift-factor must be greater than 0")

    frame = load_dataset(args.data)
    reference, _, stream = chronological_split(frame)
    feature_names = feature_columns(frame)
    reference_features = reference[feature_names]
    stream_features = stream[feature_names].copy()
    if args.shift_factor != 1.0:
        stream_features["Torque [Nm]"] *= args.shift_factor
        stream_features["Tool wear [min]"] *= args.shift_factor

    batch_reports = []
    client_latencies: list[float] = []
    service_latencies: list[float] = []
    alerts = 0
    successes = 0
    errors = 0
    prediction_records = []
    for start in range(0, len(stream_features), args.batch_size):
        stop = min(start + args.batch_size, len(stream_features))
        batch = stream_features.iloc[start:stop]
        batch_service_ms: list[float] = []
        batch_alerts = 0
        for offset, (row_index, row) in enumerate(batch.iterrows()):
            try:
                result, client_ms = call_api(args.api_url, api_payload(row))
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
                errors += 1
                print(f"request failed for stream row {start + offset}: {error}")
                continue
            client_latencies.append(client_ms)
            service_ms = float(result["inference_ms"])
            service_latencies.append(service_ms)
            batch_service_ms.append(service_ms)
            alerts += bool(result["alert"])
            batch_alerts += bool(result["alert"])
            successes += 1
            prediction_records.append(
                {
                    "UDI": int(stream.loc[row_index, "UDI"]),
                    "failure_risk": float(result["failure_risk"]),
                    "alert": bool(result["alert"]),
                    "threshold": float(result["threshold"]),
                    "model_version": str(result["model_version"]),
                    "batch_index": len(batch_reports),
                    "predicted_at": datetime.now(UTC).isoformat(),
                }
            )

        batch_drift = drift_report(reference_features, batch)
        batch_reports.append(
            {
                "batch_index": len(batch_reports),
                "rows": len(batch),
                "successful_requests": len(batch_service_ms),
                "alerts": int(batch_alerts),
                "service_latency_mean_ms": (
                    sum(batch_service_ms) / len(batch_service_ms) if batch_service_ms else None
                ),
                "drift": batch_drift,
                "drifted_features": [
                    name for name, value in batch_drift.items() if value["alert"]
                ],
            }
        )

    report = {
        "mode": "controlled_distribution_shift" if args.shift_factor != 1 else "held_out_stream",
        "shift_factor": args.shift_factor,
        "stream_rows": len(stream_features),
        "successful_requests": successes,
        "failed_requests": errors,
        "alert_count": alerts,
        "alert_rate": alerts / successes if successes else 0.0,
        "client_latency_p50_ms": (
            float(pd.Series(client_latencies).quantile(0.50)) if client_latencies else None
        ),
        "client_latency_p95_ms": (
            float(pd.Series(client_latencies).quantile(0.95)) if client_latencies else None
        ),
        "service_latency_p50_ms": (
            float(pd.Series(service_latencies).quantile(0.50)) if service_latencies else None
        ),
        "service_latency_p95_ms": (
            float(pd.Series(service_latencies).quantile(0.95)) if service_latencies else None
        ),
        "label_usage": "Labels are not sent to the API or used in this stream report.",
        "interpretation": (
            "Feature drift is a review signal, not proof that model quality has degraded. "
            "The shifted mode changes inputs synthetically; its predictions are illustrative."
        ),
        "batches": batch_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.write_text(
        "".join(json.dumps(record) + "\n" for record in prediction_records), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"saved stream report: {args.output.resolve()}")
    print(f"saved prediction log: {args.predictions_output.resolve()}")


if __name__ == "__main__":
    main()
