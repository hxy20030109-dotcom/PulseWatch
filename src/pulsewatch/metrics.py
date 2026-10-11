"""Metrics and validation-only alert-threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve


def choose_f2_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose the threshold maximizing F2 on validation labels and scores."""
    if len(labels) != len(probabilities) or len(labels) == 0:
        raise ValueError("labels and probabilities must have equal, non-zero lengths")
    if len(np.unique(labels)) < 2:
        raise ValueError("Threshold selection requires both positive and negative labels")
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        raise ValueError("No candidate thresholds were produced")
    f2 = 5 * precision[:-1] * recall[:-1] / (4 * precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f2))])


def classification_report(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Summarize imbalanced binary classification at a fixed threshold."""
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    return {
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "precision": float(precision),
        "recall": float(recall),
        "f2": float(f2),
        "threshold": float(threshold),
        "alert_rate": float(predicted.mean()),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
