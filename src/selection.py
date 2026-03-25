from __future__ import annotations


def parameter_count(input_dim: int, hidden_layers: tuple[int, ...]) -> int:
    total = 0
    previous_dim = input_dim
    for layer_dim in (*hidden_layers, 1):
        total += (previous_dim + 1) * layer_dim
        previous_dim = layer_dim
    return total


def select_best_manual_model(results: list[dict]) -> dict:
    return sorted(
        results,
        key=lambda item: (
            -item["metrics"]["accuracy"],
            item["config"]["n_steps"],
            item["metrics"]["parameter_count"],
            -item["metrics"]["validation_accuracy"],
        ),
    )[0]
