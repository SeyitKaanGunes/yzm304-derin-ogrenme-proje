from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


class ManualMLPClassifier:
    def __init__(
        self,
        input_dim: int,
        hidden_layers: tuple[int, ...] = (6,),
        learning_rate: float = 0.01,
        l2_lambda: float = 0.0,
        seed: int = 42,
        batch_size: int | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.seed = seed
        self.batch_size = batch_size
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.history = TrainingHistory()
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        rng = np.random.default_rng(self.seed)
        layer_dims = (self.input_dim, *self.hidden_layers, 1)
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(layer_dims[:-1], layer_dims[1:]):
            weight = rng.normal(loc=0.0, scale=0.01, size=(fan_out, fan_in))
            bias = np.zeros((fan_out, 1))
            self.weights.append(weight)
            self.biases.append(bias)

    def _forward(self, X: np.ndarray) -> list[np.ndarray]:
        activations = [X.T]
        for weight, bias in zip(self.weights, self.biases):
            z = weight @ activations[-1] + bias
            activations.append(sigmoid(z))
        return activations

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        sample_count = y_true.shape[0]
        y_true_t = y_true.T
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        bce = -np.sum(y_true_t * np.log(y_pred) + (1 - y_true_t) * np.log(1 - y_pred)) / sample_count
        l2_penalty = 0.0
        if self.l2_lambda > 0:
            l2_penalty = sum(np.sum(weight**2) for weight in self.weights) * (self.l2_lambda / (2 * sample_count))
        return float(bce + l2_penalty)

    def _backward(self, y_true: np.ndarray, activations: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        sample_count = y_true.shape[0]
        y_true_t = y_true.T

        grad_weights = [np.zeros_like(weight) for weight in self.weights]
        grad_biases = [np.zeros_like(bias) for bias in self.biases]
        delta = activations[-1] - y_true_t

        for layer_index in reversed(range(len(self.weights))):
            previous_activation = activations[layer_index]
            grad_weights[layer_index] = (delta @ previous_activation.T) / sample_count
            if self.l2_lambda > 0:
                grad_weights[layer_index] += (self.l2_lambda / sample_count) * self.weights[layer_index]
            grad_biases[layer_index] = np.sum(delta, axis=1, keepdims=True) / sample_count

            if layer_index > 0:
                propagated = self.weights[layer_index].T @ delta
                current_activation = activations[layer_index]
                delta = propagated * current_activation * (1 - current_activation)

        return grad_weights, grad_biases

    def _apply_gradients(self, grad_weights: list[np.ndarray], grad_biases: list[np.ndarray]) -> None:
        for layer_index in range(len(self.weights)):
            self.weights[layer_index] -= self.learning_rate * grad_weights[layer_index]
            self.biases[layer_index] -= self.learning_rate * grad_biases[layer_index]

    def _iter_batches(self, X: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.batch_size is None or self.batch_size >= len(X):
            return [(X, y)]

        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(X))
        batches: list[tuple[np.ndarray, np.ndarray]] = []
        for start in range(0, len(X), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            batches.append((X[batch_indices], y[batch_indices]))
        return batches

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations = self._forward(X)
        return activations[-1].T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return float(np.mean(predictions == y))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_steps: int,
    ) -> TrainingHistory:
        self.history = TrainingHistory()
        for _ in range(n_steps):
            for X_batch, y_batch in self._iter_batches(X_train, y_train):
                activations = self._forward(X_batch)
                grad_weights, grad_biases = self._backward(y_batch, activations)
                self._apply_gradients(grad_weights, grad_biases)

            train_probabilities = self.predict_proba(X_train).T
            validation_probabilities = self.predict_proba(X_val).T
            self.history.train_loss.append(self._compute_loss(y_train, train_probabilities))
            self.history.val_loss.append(self._compute_loss(y_val, validation_probabilities))
            self.history.train_accuracy.append(self.score(X_train, y_train))
            self.history.val_accuracy.append(self.score(X_val, y_val))

        return self.history
