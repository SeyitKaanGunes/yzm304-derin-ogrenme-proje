from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data import RAW_DATA_PATH, SPLIT_MANIFEST_PATH, export_raw_dataset, prepare_dataset


def test_dataset_split_shapes_and_assets_exist() -> None:
    export_raw_dataset()
    split = prepare_dataset(standardize=False)

    assert RAW_DATA_PATH.exists()
    assert split.X_train.shape == (341, 30)
    assert split.X_val.shape == (114, 30)
    assert split.X_test.shape == (114, 30)
    assert split.y_train.shape == (341, 1)
    assert split.y_val.shape == (114, 1)
    assert split.y_test.shape == (114, 1)


def test_standardization_centers_training_features() -> None:
    split = prepare_dataset(standardize=True)
    assert np.allclose(split.X_train.mean(axis=0), 0.0, atol=1e-7)


def test_split_manifest_matches_expected_sizes() -> None:
    prepare_dataset(standardize=True)
    payload = json.loads(Path(SPLIT_MANIFEST_PATH).read_text(encoding="utf-8"))
    assert payload["train_size"] == 341
    assert payload["validation_size_count"] == 114
    assert payload["test_size_count"] == 114
    assert len(payload["train_ids"]) == 341
