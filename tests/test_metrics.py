from __future__ import annotations

import numpy as np

from src.metrics import evaluate_predictions


def test_metrics_dictionary_contains_core_keys() -> None:
    y_true = np.array([[0], [1], [1], [0]])
    y_pred = np.array([[0], [1], [0], [0]])
    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert "classification_report" in metrics
    assert "balanced_accuracy" in metrics
