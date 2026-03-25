from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import evaluate_predictions


def save_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_markdown(markdown_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_text, encoding="utf-8")


def save_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_training_curves(
    train_loss: list[float],
    val_loss: list[float],
    train_accuracy: list[float],
    val_accuracy: list[float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Validation Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Binary Cross Entropy")
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, label="Train Accuracy")
    axes[1].plot(epochs, val_accuracy, label="Validation Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(conf_matrix: list[list[int]], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(conf_matrix)

    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, str(matrix[row_index, col_index]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
