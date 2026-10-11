"""Train baseline failure-risk classifiers and write a held-out evaluation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pulsewatch.data import TARGET, chronological_split, feature_columns, load_dataset
from pulsewatch.metrics import choose_f2_threshold, classification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/ai4i2020.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_pipeline(frame, estimator) -> Pipeline:
    inputs = feature_columns(frame)
    categorical = frame[inputs].select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in inputs if column not in categorical]
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def main() -> None:
    args = parse_args()
    frame = load_dataset(args.data)
    train, validation, test = chronological_split(frame)
    inputs = feature_columns(frame)
    x_train, y_train = train[inputs], train[TARGET].astype(int)
    x_validation, y_validation = validation[inputs], validation[TARGET].astype(int)
    x_test, y_test = test[inputs], test[TARGET].astype(int)

    estimators = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=args.seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.seed,
        ),
    }
    results = {
        "dataset": "UCI AI4I 2020 Predictive Maintenance Dataset",
        "data_is_synthetic": True,
        "split_policy": "ordered rows: 70% train, 15% validation, 15% test",
        "seed": args.seed,
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "features": inputs,
        "threshold_selection": "maximize F2 on validation set only",
        "models": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, estimator in estimators.items():
        pipeline = build_pipeline(frame, estimator)
        pipeline.fit(x_train, y_train)
        validation_probabilities = pipeline.predict_proba(x_validation)[:, 1]
        threshold = choose_f2_threshold(y_validation.to_numpy(), validation_probabilities)
        test_probabilities = pipeline.predict_proba(x_test)[:, 1]
        results["models"][name] = {
            "validation": classification_report(
                y_validation.to_numpy(), validation_probabilities, threshold
            ),
            "test": classification_report(y_test.to_numpy(), test_probabilities, threshold),
        }
        joblib.dump(
            {
                "pipeline": pipeline,
                "threshold": threshold,
                "feature_columns": inputs,
                "model_version": "0.1.0",
                "model_name": name,
            },
            args.output_dir / f"{name}.joblib",
        )

    report_path = args.output_dir / "baseline_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"saved report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
