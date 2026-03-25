from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, TARGET_NAME, TEST_SIZE, VAL_SIZE


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "breast_cancer.csv"
SPLIT_MANIFEST_PATH = DATA_DIR / "splits" / "split_manifest.json"


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_name: str
    standardize: bool
    train_ids: list[int]
    val_ids: list[int]
    test_ids: list[int]


def export_raw_dataset(force: bool = False) -> Path:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if force or not RAW_DATA_PATH.exists():
        dataset = load_breast_cancer(as_frame=True)
        frame = dataset.frame.copy()
        frame.to_csv(RAW_DATA_PATH, index=False)
    return RAW_DATA_PATH


def load_raw_breast_cancer_dataframe() -> pd.DataFrame:
    export_raw_dataset()
    return pd.read_csv(RAW_DATA_PATH)


def _load_shuffled_dataframe_with_ids(random_state: int = RANDOM_STATE) -> pd.DataFrame:
    frame = load_raw_breast_cancer_dataframe().copy()
    frame["sample_id"] = np.arange(len(frame))
    frame = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return frame


def load_breast_cancer_dataframe(random_state: int = RANDOM_STATE) -> pd.DataFrame:
    frame = _load_shuffled_dataframe_with_ids(random_state=random_state)
    return frame.drop(columns=["sample_id"])


def _write_split_manifest(split: DatasetSplit, random_state: int, test_size: float, val_size: float) -> None:
    SPLIT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "breast_cancer",
        "random_state": random_state,
        "test_size": test_size,
        "validation_size": val_size,
        "train_size": len(split.train_ids),
        "validation_size_count": len(split.val_ids),
        "test_size_count": len(split.test_ids),
        "train_ids": split.train_ids,
        "validation_ids": split.val_ids,
        "test_ids": split.test_ids,
    }
    SPLIT_MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_dataset(
    *,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    standardize: bool,
) -> DatasetSplit:
    frame = _load_shuffled_dataframe_with_ids(random_state=random_state)
    feature_names = [column for column in frame.columns if column not in {TARGET_NAME, "sample_id"}]

    X = frame[feature_names].to_numpy(dtype=np.float64)
    y = frame[TARGET_NAME].to_numpy(dtype=np.float64).reshape(-1, 1)
    ids = frame["sample_id"].to_numpy(dtype=np.int64)

    X_train_full, X_test, y_train_full, y_test, ids_train_full, ids_test = train_test_split(
        X,
        y,
        ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_train_full,
        y_train_full,
        ids_train_full,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    split = DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        target_name=TARGET_NAME,
        standardize=standardize,
        train_ids=ids_train.tolist(),
        val_ids=ids_val.tolist(),
        test_ids=ids_test.tolist(),
    )
    _write_split_manifest(split, random_state=random_state, test_size=test_size, val_size=val_size)
    return split
