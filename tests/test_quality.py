import pandas as pd
import pytest

from pulsewatch.quality import evaluate_delayed_labels


def test_delayed_labels_join_by_udi_and_calculate_quality():
    predictions = pd.DataFrame(
        {
            "UDI": [10, 20, 30, 40],
            "failure_risk": [0.1, 0.8, 0.2, 0.9],
            "threshold": [0.5] * 4,
            "model_version": ["0.1.0"] * 4,
        }
    )
    labels = pd.DataFrame({"UDI": [40, 20, 10, 30], "Machine failure": [1, 1, 0, 0]})

    report = evaluate_delayed_labels(predictions, labels)

    assert report["status"] == "complete"
    assert report["labeled_count"] == 4
    assert report["pending_label_count"] == 0
    assert report["metrics"]["pr_auc"] == 1.0
    assert report["metrics"]["true_positives"] == 2


def test_missing_labels_are_reported_as_pending_without_fake_metrics():
    predictions = pd.DataFrame(
        {
            "UDI": [1, 2],
            "failure_risk": [0.2, 0.7],
            "threshold": [0.5, 0.5],
            "model_version": ["0.1.0", "0.1.0"],
        }
    )
    labels = pd.DataFrame({"UDI": [1], "Machine failure": [0]})

    report = evaluate_delayed_labels(predictions, labels)

    assert report["status"] == "partial"
    assert report["labeled_count"] == 1
    assert report["pending_label_count"] == 1
    assert report["metrics"] is None


@pytest.mark.parametrize(
    "predictions",
    [
        pd.DataFrame(
            {
                "UDI": [1, 1],
                "failure_risk": [0.2, 0.3],
                "threshold": [0.5, 0.5],
                "model_version": ["0.1.0", "0.1.0"],
            }
        ),
        pd.DataFrame(
            {
                "UDI": [1, 2],
                "failure_risk": [0.2, 0.3],
                "threshold": [0.5, 0.6],
                "model_version": ["0.1.0", "0.1.0"],
            }
        ),
    ],
)
def test_reject_duplicate_events_and_mixed_thresholds(predictions):
    labels = pd.DataFrame({"UDI": [1, 2], "Machine failure": [0, 1]})
    with pytest.raises(ValueError):
        evaluate_delayed_labels(predictions, labels)
