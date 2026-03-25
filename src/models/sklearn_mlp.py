from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier


@dataclass
class SklearnExperimentResult:
    model: MLPClassifier
    train_loss_curve: list[float]


def train_sklearn_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_layers: tuple[int, ...],
    learning_rate: float,
    alpha: float,
    batch_size: int,
    max_iter: int,
    random_state: int,
) -> SklearnExperimentResult:
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="logistic",
        solver="sgd",
        learning_rate="constant",
        learning_rate_init=learning_rate,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=max_iter,
        shuffle=False,
        random_state=random_state,
        momentum=0.0,
        n_iter_no_change=max_iter + 1,
        tol=0.0,
    )
    model.fit(X_train, y_train.ravel())
    return SklearnExperimentResult(model=model, train_loss_curve=list(model.loss_curve_))
