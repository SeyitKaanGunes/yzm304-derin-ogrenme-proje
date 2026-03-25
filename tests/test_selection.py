from __future__ import annotations

from src.selection import select_best_manual_model


def test_selection_prefers_higher_accuracy_then_lower_steps() -> None:
    low = {
        "config": {"n_steps": 1000},
        "metrics": {"accuracy": 0.90, "parameter_count": 10, "validation_accuracy": 0.92},
    }
    high = {
        "config": {"n_steps": 1500},
        "metrics": {"accuracy": 0.93, "parameter_count": 15, "validation_accuracy": 0.91},
    }
    chosen = select_best_manual_model([low, high])
    assert chosen is high
