import numpy as np
import pytest

from pulsewatch.metrics import choose_f2_threshold, classification_report


def test_threshold_is_selected_from_scores_and_report_counts_are_consistent() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.6, 0.9])

    threshold = choose_f2_threshold(labels, probabilities)
    report = classification_report(labels, probabilities, threshold)

    assert threshold in probabilities
    assert report["true_negatives"] + report["false_positives"] == 2
    assert report["false_negatives"] + report["true_positives"] == 2
    assert report["recall"] == 1.0


def test_threshold_selection_rejects_single_class_validation_data() -> None:
    with pytest.raises(ValueError, match="both positive and negative"):
        choose_f2_threshold(np.array([0, 0]), np.array([0.1, 0.2]))


def test_report_handles_no_positive_predictions() -> None:
    report = classification_report(
        np.array([0, 1]), np.array([0.1, 0.2]), threshold=0.9
    )

    assert report["precision"] == 0.0
    assert report["recall"] == 0.0
    assert report["true_negatives"] == 1
    assert report["false_negatives"] == 1
