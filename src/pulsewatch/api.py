"""HTTP inference API for the saved PulseWatch model."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("pulsewatch.inference")


class DeviceReading(BaseModel):
    """One AI4I operating record, using API-friendly field names."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    type_: Annotated[str, Field(alias="type", pattern="^[LMH]$")]
    air_temperature_k: Annotated[float, Field(alias="air_temperature_k", gt=0)]
    process_temperature_k: Annotated[float, Field(alias="process_temperature_k", gt=0)]
    rotational_speed_rpm: Annotated[int, Field(alias="rotational_speed_rpm", gt=0)]
    torque_nm: Annotated[float, Field(alias="torque_nm", ge=0)]
    tool_wear_min: Annotated[int, Field(alias="tool_wear_min", ge=0)]

    def model_features(self) -> dict[str, str | float | int]:
        """Map validated API fields to the training column names."""
        return {
            "Type": self.type_,
            "Air temperature [K]": self.air_temperature_k,
            "Process temperature [K]": self.process_temperature_k,
            "Rotational speed [rpm]": self.rotational_speed_rpm,
            "Torque [Nm]": self.torque_nm,
            "Tool wear [min]": self.tool_wear_min,
        }


class PredictionResponse(BaseModel):
    failure_risk: float
    alert: bool
    threshold: float
    model_name: str
    model_version: str
    inference_ms: float


app = FastAPI(title="PulseWatch API", version="0.1.0")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_artifact(path_string: str) -> dict:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = joblib.load(path)
    if not {"pipeline", "threshold"} <= artifact.keys():
        raise ValueError(f"Invalid PulseWatch artifact: {path}")
    return artifact


def artifact_path() -> str:
    return os.environ.get("PULSEWATCH_MODEL_PATH", "artifacts/random_forest.joblib")


def read_report(name: str) -> dict | None:
    path = PROJECT_ROOT / "artifacts" / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail=f"Could not read {name}") from error


@app.get("/dashboard", include_in_schema=False)
def dashboard() -> FileResponse:
    page = Path(__file__).with_name("static") / "dashboard.html"
    if not page.is_file():
        raise HTTPException(status_code=503, detail="Dashboard page is unavailable")
    return FileResponse(page, media_type="text/html")


@app.get("/monitoring/summary")
def monitoring_summary() -> dict:
    stream = read_report("stream_report.json")
    baseline = read_report("baseline_report.json")
    delayed_quality = read_report("delayed_quality_report.json")
    if stream is None:
        return {
            "available": False,
            "message": "Run scripts/simulate_stream.py to create the first stream report.",
            "baseline_available": baseline is not None,
            "delayed_quality_available": delayed_quality is not None,
        }

    stream_path = PROJECT_ROOT / "artifacts" / "stream_report.json"
    modified_at = datetime.fromtimestamp(stream_path.stat().st_mtime, tz=UTC).isoformat()
    feature_scores: dict[str, dict] = {}
    batches = stream.get("batches", [])
    for batch in batches:
        for name, value in batch.get("drift", {}).items():
            item = feature_scores.setdefault(
                name,
                {"metric": value.get("metric", "unknown"), "scores": [], "alert_batches": 0},
            )
            item["scores"].append(float(value.get("score", 0.0)))
            item["alert_batches"] += int(bool(value.get("alert")))

    drift = [
        {
            "feature": name,
            "metric": item["metric"],
            "mean_score": sum(item["scores"]) / len(item["scores"]),
            "max_score": max(item["scores"]),
            "alert_batches": item["alert_batches"],
        }
        for name, item in feature_scores.items()
        if item["scores"]
    ]
    random_forest = (baseline or {}).get("models", {}).get("random_forest", {}).get("test", {})
    return {
        "available": True,
        "generated_at": modified_at,
        "run": {
            "mode": stream.get("mode", "unknown"),
            "shift_factor": stream.get("shift_factor", 1.0),
            "stream_rows": stream.get("stream_rows", 0),
            "successful_requests": stream.get("successful_requests", 0),
            "failed_requests": stream.get("failed_requests", 0),
            "alert_count": stream.get("alert_count", 0),
            "alert_rate": stream.get("alert_rate", 0.0),
        },
        "latency_ms": {
            "client_p50": stream.get("client_latency_p50_ms"),
            "client_p95": stream.get("client_latency_p95_ms"),
            "service_p50": stream.get("service_latency_p50_ms"),
            "service_p95": stream.get("service_latency_p95_ms"),
        },
        "quality": {
            "model": "Random Forest",
            "pr_auc": random_forest.get("pr_auc"),
            "precision": random_forest.get("precision"),
            "recall": random_forest.get("recall"),
            "f2": random_forest.get("f2"),
        },
        "delayed_quality": {
            "available": delayed_quality is not None,
            "status": (delayed_quality or {}).get("status"),
            "model_version": (delayed_quality or {}).get("model_version"),
            "labeled_count": (delayed_quality or {}).get("labeled_count", 0),
            "pending_label_count": (delayed_quality or {}).get("pending_label_count", 0),
            "metrics": (delayed_quality or {}).get("metrics"),
            "note": (delayed_quality or {}).get("note"),
            "generated_at": (delayed_quality or {}).get("generated_at"),
        },
        "drift": drift,
        "batches": [
            {
                "index": batch.get("batch_index", index),
                "alert_rate": (
                    batch.get("alerts", 0) / batch.get("successful_requests", 1)
                    if batch.get("successful_requests", 0)
                    else 0.0
                ),
                "drifted_features": len(batch.get("drifted_features", [])),
                "requests": batch.get("successful_requests", 0),
            }
            for index, batch in enumerate(batches)
        ],
        "interpretation": stream.get(
            "interpretation",
            "Feature drift is a review signal, not proof that model quality has degraded.",
        ),
    }


@app.get("/health")
def health() -> dict[str, str]:
    try:
        artifact = load_artifact(artifact_path())
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail="Model artifact is unavailable") from error
    return {
        "status": "ready",
        "model_name": artifact.get("model_name", "unknown"),
        "model_version": artifact.get("model_version", "unknown"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: DeviceReading) -> PredictionResponse:
    try:
        artifact = load_artifact(artifact_path())
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail="Model artifact is unavailable") from error

    started = time.perf_counter()
    features = pd.DataFrame([reading.model_features()])
    risk = float(artifact["pipeline"].predict_proba(features)[0, 1])
    elapsed_ms = (time.perf_counter() - started) * 1000
    threshold = float(artifact["threshold"])
    alert = risk >= threshold
    logger.info(
        "prediction model=%s version=%s risk=%.6f alert=%s latency_ms=%.3f",
        artifact.get("model_name", "unknown"),
        artifact.get("model_version", "unknown"),
        risk,
        alert,
        elapsed_ms,
    )
    return PredictionResponse(
        failure_risk=risk,
        alert=alert,
        threshold=threshold,
        model_name=artifact.get("model_name", "unknown"),
        model_version=artifact.get("model_version", "unknown"),
        inference_ms=elapsed_ms,
    )
