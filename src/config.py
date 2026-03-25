from __future__ import annotations

from dataclasses import dataclass


RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
TARGET_NAME = "target"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    backend: str
    standardize: bool
    hidden_layers: tuple[int, ...]
    n_steps: int
    learning_rate: float
    l2_lambda: float
    batch_size: int | None = None


MANUAL_EXPERIMENTS = [
    ExperimentConfig(
        name="manual_lab_baseline",
        backend="numpy",
        standardize=False,
        hidden_layers=(6,),
        n_steps=500,
        learning_rate=0.01,
        l2_lambda=0.0,
    ),
    ExperimentConfig(
        name="manual_improved_baseline",
        backend="numpy",
        standardize=True,
        hidden_layers=(6,),
        n_steps=1000,
        learning_rate=0.05,
        l2_lambda=0.0,
    ),
    ExperimentConfig(
        name="manual_deeper_regularized",
        backend="numpy",
        standardize=True,
        hidden_layers=(12, 6),
        n_steps=1500,
        learning_rate=0.05,
        l2_lambda=0.001,
        batch_size=32,
    ),
]


SKLEARN_EXPERIMENTS = [
    ExperimentConfig(
        name="sklearn_baseline",
        backend="sklearn",
        standardize=True,
        hidden_layers=(6,),
        n_steps=1000,
        learning_rate=0.05,
        l2_lambda=0.0,
    ),
    ExperimentConfig(
        name="sklearn_deeper_regularized",
        backend="sklearn",
        standardize=True,
        hidden_layers=(12, 6),
        n_steps=1500,
        learning_rate=0.05,
        l2_lambda=0.001,
        batch_size=32,
    ),
]
