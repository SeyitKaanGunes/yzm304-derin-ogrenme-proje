from __future__ import annotations

from src.data import prepare_dataset
from src.metrics import evaluate_predictions
from src.models.sklearn_mlp import train_sklearn_mlp


def test_sklearn_backend_reaches_reasonable_accuracy() -> None:
    split = prepare_dataset(standardize=True)
    result = train_sklearn_mlp(
        split.X_train,
        split.y_train,
        hidden_layers=(6,),
        learning_rate=0.05,
        alpha=0.0,
        batch_size=len(split.X_train),
        max_iter=200,
        random_state=42,
    )
    metrics = evaluate_predictions(split.y_test, result.model.predict(split.X_test).reshape(-1, 1))
    assert metrics["accuracy"] >= 0.88
