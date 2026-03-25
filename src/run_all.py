from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from src.config import MANUAL_EXPERIMENTS, RANDOM_STATE, SKLEARN_EXPERIMENTS, ExperimentConfig
from src.data import export_raw_dataset, prepare_dataset
from src.metrics import evaluate_predictions
from src.models.manual_mlp import ManualMLPClassifier
from src.models.sklearn_mlp import train_sklearn_mlp
from src.reporting import save_confusion_matrix, save_json, save_markdown, save_table, save_training_curves
from src.selection import parameter_count, select_best_manual_model


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES_DIR = OUTPUTS / "figures"
TABLES_DIR = OUTPUTS / "tables"
REPORTS_DIR = OUTPUTS / "reports"


def _ensure_output_directories() -> None:
    for path in (FIGURES_DIR, TABLES_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _manual_result(config: ExperimentConfig) -> dict:
    dataset = prepare_dataset(standardize=config.standardize)
    model = ManualMLPClassifier(
        input_dim=dataset.X_train.shape[1],
        hidden_layers=config.hidden_layers,
        learning_rate=config.learning_rate,
        l2_lambda=config.l2_lambda,
        seed=RANDOM_STATE,
        batch_size=config.batch_size,
    )
    history = model.fit(
        dataset.X_train,
        dataset.y_train,
        X_val=dataset.X_val,
        y_val=dataset.y_val,
        n_steps=config.n_steps,
    )
    metrics = evaluate_predictions(dataset.y_test, model.predict(dataset.X_test))
    metrics["validation_accuracy"] = history.val_accuracy[-1]
    metrics["train_accuracy"] = history.train_accuracy[-1]
    metrics["n_steps"] = config.n_steps
    metrics["standardize"] = config.standardize
    metrics["hidden_layers"] = list(config.hidden_layers)
    metrics["l2_lambda"] = config.l2_lambda
    metrics["parameter_count"] = parameter_count(dataset.X_train.shape[1], config.hidden_layers)

    save_training_curves(
        history.train_loss,
        history.val_loss,
        history.train_accuracy,
        history.val_accuracy,
        FIGURES_DIR / f"{config.name}_curves.png",
    )
    save_confusion_matrix(
        metrics["confusion_matrix"],
        FIGURES_DIR / f"{config.name}_confusion_matrix.png",
        title=config.name,
    )
    return {"config": asdict(config), "metrics": metrics}


def _sklearn_result(config: ExperimentConfig) -> dict:
    dataset = prepare_dataset(standardize=config.standardize)
    result = train_sklearn_mlp(
        dataset.X_train,
        dataset.y_train,
        hidden_layers=config.hidden_layers,
        learning_rate=config.learning_rate,
        alpha=config.l2_lambda,
        batch_size=config.batch_size or len(dataset.X_train),
        max_iter=config.n_steps,
        random_state=RANDOM_STATE,
    )
    predictions = result.model.predict(dataset.X_test).reshape(-1, 1)
    metrics = evaluate_predictions(dataset.y_test, predictions)
    metrics["train_loss_curve_length"] = len(result.train_loss_curve)
    metrics["n_steps"] = config.n_steps
    metrics["standardize"] = config.standardize
    metrics["hidden_layers"] = list(config.hidden_layers)
    metrics["l2_lambda"] = config.l2_lambda
    metrics["parameter_count"] = parameter_count(dataset.X_train.shape[1], config.hidden_layers)
    save_confusion_matrix(
        metrics["confusion_matrix"],
        FIGURES_DIR / f"{config.name}_confusion_matrix.png",
        title=config.name,
    )
    return {"config": asdict(config), "metrics": metrics}


def _selection_rows(manual_results: list[dict]) -> list[dict]:
    rows = []
    for result in sorted(
        manual_results,
        key=lambda item: (-item["metrics"]["accuracy"], item["config"]["n_steps"], item["metrics"]["parameter_count"]),
    ):
        rows.append(
            {
                "model": result["config"]["name"],
                "backend": result["config"]["backend"],
                "accuracy": result["metrics"]["accuracy"],
                "balanced_accuracy": result["metrics"]["balanced_accuracy"],
                "precision": result["metrics"]["precision"],
                "recall": result["metrics"]["recall"],
                "f1": result["metrics"]["f1"],
                "validation_accuracy": result["metrics"]["validation_accuracy"],
                "train_accuracy": result["metrics"]["train_accuracy"],
                "n_steps": result["config"]["n_steps"],
                "hidden_layers": "-".join(str(value) for value in result["config"]["hidden_layers"]),
                "standardize": result["config"]["standardize"],
                "batch_size": result["config"]["batch_size"] or "full",
                "l2_lambda": result["config"]["l2_lambda"],
                "parameter_count": result["metrics"]["parameter_count"],
            }
        )
    return rows


def _backend_rows(selected_manual_model: dict, sklearn_results: list[dict]) -> list[dict]:
    rows = [
        {
            "model": selected_manual_model["config"]["name"],
            "backend": selected_manual_model["config"]["backend"],
            "accuracy": selected_manual_model["metrics"]["accuracy"],
            "balanced_accuracy": selected_manual_model["metrics"]["balanced_accuracy"],
            "precision": selected_manual_model["metrics"]["precision"],
            "recall": selected_manual_model["metrics"]["recall"],
            "f1": selected_manual_model["metrics"]["f1"],
        }
    ]
    for result in sklearn_results:
        rows.append(
            {
                "model": result["config"]["name"],
                "backend": result["config"]["backend"],
                "accuracy": result["metrics"]["accuracy"],
                "balanced_accuracy": result["metrics"]["balanced_accuracy"],
                "precision": result["metrics"]["precision"],
                "recall": result["metrics"]["recall"],
                "f1": result["metrics"]["f1"],
            }
        )
    return rows


def _selected_model_report(selected_manual_model: dict) -> str:
    metrics = selected_manual_model["metrics"]
    config = selected_manual_model["config"]
    return f"""# Selected Model Report

Selected model: `{config['name']}`

- Backend: `{config['backend']}`
- Hidden layers: `{tuple(config['hidden_layers'])}`
- Learning rate: `{config['learning_rate']}`
- Steps: `{config['n_steps']}`
- L2 lambda: `{config['l2_lambda']}`
- Batch size: `{config['batch_size'] or 'full'}`
- Accuracy: `{metrics['accuracy']:.4f}`
- Balanced accuracy: `{metrics['balanced_accuracy']:.4f}`
- Precision: `{metrics['precision']:.4f}`
- Recall: `{metrics['recall']:.4f}`
- F1: `{metrics['f1']:.4f}`
- Validation accuracy: `{metrics['validation_accuracy']:.4f}`
- Train accuracy: `{metrics['train_accuracy']:.4f}`

Related artifacts:

- `outputs/figures/{config['name']}_curves.png`
- `outputs/figures/{config['name']}_confusion_matrix.png`
- `outputs/tables/model_selection.csv`
- `outputs/tables/backend_comparison_metrics.csv`
"""


def main() -> None:
    export_raw_dataset()
    _ensure_output_directories()

    manual_results = [_manual_result(config) for config in MANUAL_EXPERIMENTS]
    sklearn_results = [_sklearn_result(config) for config in SKLEARN_EXPERIMENTS]
    selected_manual_model = select_best_manual_model(manual_results)

    results = {
        "manual_experiments": manual_results,
        "sklearn_experiments": sklearn_results,
        "selected_manual_model": selected_manual_model,
    }

    save_json(results, OUTPUTS / "experiment_results.json")
    save_table(_selection_rows(manual_results), TABLES_DIR / "model_selection.csv")
    save_table(_backend_rows(selected_manual_model, sklearn_results), TABLES_DIR / "backend_comparison_metrics.csv")
    save_markdown(_selected_model_report(selected_manual_model), REPORTS_DIR / "selected_model_report.md")


if __name__ == "__main__":
    main()
