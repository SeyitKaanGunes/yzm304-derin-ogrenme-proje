from __future__ import annotations

from src.data import prepare_dataset
from src.models.manual_mlp import ManualMLPClassifier


def test_manual_mlp_reaches_reasonable_accuracy() -> None:
    split = prepare_dataset(standardize=True)
    model = ManualMLPClassifier(
        input_dim=split.X_train.shape[1],
        hidden_layers=(6,),
        learning_rate=0.1,
        l2_lambda=0.0,
        seed=42,
    )
    history = model.fit(
        split.X_train,
        split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        n_steps=500,
    )

    assert history.train_loss[-1] < history.train_loss[0]
    assert model.score(split.X_test, split.y_test) >= 0.90
