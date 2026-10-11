# PulseWatch

**Predictive Maintenance & Model Monitoring**

PulseWatch is a small, reproducible machine-learning service prototype that
estimates machine-failure risk from operating measurements and demonstrates how
to monitor predictions and changing input data.

The first milestone uses the UCI AI4I 2020 Predictive Maintenance Dataset. UCI
describes it as synthetic data; results from it are a reproducible prototype,
not evidence of performance in a real factory. Dataset citation, license, and
limitations are recorded in [`data/README.md`](data/README.md).

## Implemented so far

- Chronological train/validation/test split using the dataset's row identifier.
- Logistic regression and random-forest baselines.
- Precision-recall metrics for imbalanced failure labels.
- Alert-threshold selection on validation data only.
- A saved model and machine-readable evaluation report.
- FastAPI `/health` and `/predict` endpoints with input checks, model version,
  risk score, and inference time.
- Numeric PSI and categorical distribution comparisons for labeled or unlabeled
  input batches.

The stream simulator, service-level summary, dashboard, and container recipe
are included. Drift alerts are signals for investigation; drift alone does not
prove that predictive quality fell. See [ARCHITECTURE.md](ARCHITECTURE.md) for
the data flow and boundaries.

## Start

Use Python 3.12. Download the CSV from the official
[UCI dataset page](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bda)
and place it at `data/raw/ai4i2020.csv` (the local data directory is ignored by
Git).

```powershell
uv sync --extra dev
uv run python scripts/train_baselines.py --data data/raw/ai4i2020.csv
uv sync --extra service
uv run uvicorn pulsewatch.api:app --reload
```

The command writes model artifacts and a JSON report under `artifacts/`.
The API loads `artifacts/random_forest.joblib` by default; set
`PULSEWATCH_MODEL_PATH` to use another saved artifact.
Open [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard) to see
the monitoring dashboard. It summarizes the latest stream report, drift by
feature, request latency, and held-out baseline metrics. The page refreshes the
report every 15 seconds; it does not collect or retain live request history.

## Run with Docker

First download the dataset and train the model using the steps above. The model
artifact is deliberately kept outside the image and mounted from the local
`artifacts/` folder:

```powershell
docker compose up --build
```

Then check `http://127.0.0.1:8000/health` and open the dashboard URL above. The
container serves inference only; run training and the stream simulator on the
host as described below. Stop it with `Ctrl+C` (or `docker compose down`).

In a second terminal, simulate label-free requests from the held-out rows:

```powershell
uv run python scripts/simulate_stream.py --data data/raw/ai4i2020.csv
```

After the simulator finishes, refresh the dashboard to view the new report.
The summary data is also available at `/monitoring/summary`.

To see a controlled feature-drift alert, multiply torque and tool wear in the
simulated stream. The injected-shift results are illustrative and are not
quality measurements:

```powershell
uv run python scripts/simulate_stream.py --data data/raw/ai4i2020.csv --shift-factor 1.3
```

## Evaluation principles

- Keep identifiers and failure-mode target columns out of model features.
- Select the operating threshold using validation data; inspect the test set
  once after the model and threshold are fixed.
- Report PR-AUC, precision, recall, F2, confusion counts, and the selected
  threshold. Accuracy alone is not useful for this imbalanced task.
- Compare alert trade-offs and inspect false positives and false negatives.
- Record split policy, seed, package versions, and data provenance.

## Project status

Implemented: reproducible data split, baseline training and threshold
selection, FastAPI inference, feature drift and service metrics, stream
simulator, dashboard, Docker recipe, and architecture/limitations guide.

Not implemented: delayed-label ingestion and post-deployment quality tracking.
The current project evaluates held-out baseline labels offline; simulated
stream requests intentionally contain no labels. This distinction is visible
in the dashboard and stream report.

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the project boundary and
acceptance criteria.
