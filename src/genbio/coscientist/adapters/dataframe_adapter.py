"""Adapter for pandas DataFrame datasets (sequences and tabular)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genbio.coscientist.adapters.base import DatasetAdapter
from genbio.coscientist.config import DataProfile, DataType, TaskType


_LABEL_CANDIDATES = ["labels", "label", "target", "y"]
_SEQUENCE_CANDIDATES = ["sequence", "sequences", "seq", "dna", "rna", "protein"]
_EXCLUDE_COLS = {"fold_id", "fold", "split", "index"}


def _detect_label_col(df: pd.DataFrame) -> str:
    for c in _LABEL_CANDIDATES:
        if c in df.columns:
            return c
    non_feature = [c for c in df.columns if c not in _EXCLUDE_COLS]
    if len(non_feature) == 1:
        return non_feature[0]
    raise ValueError(
        f"Cannot auto-detect label column. Columns: {list(df.columns)}. "
        f"Expected one of {_LABEL_CANDIDATES}."
    )


def _detect_sequence_col(df: pd.DataFrame) -> str | None:
    for c in _SEQUENCE_CANDIDATES:
        if c in df.columns and df[c].dtype == object:
            return c
    return None


def _infer_task_type(labels: np.ndarray) -> TaskType:
    if labels.dtype.kind == "f":
        nunique = len(np.unique(labels[~np.isnan(labels)]))
        if nunique > 20:
            return TaskType.REGRESSION
    nunique = len(np.unique(labels))
    if nunique <= 30:
        return TaskType.CLASSIFICATION
    return TaskType.REGRESSION


class DataFrameAdapter(DatasetAdapter):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        self.label_col = _detect_label_col(train)
        self.seq_col = _detect_sequence_col(train)
        self._feature_cols: list[str] | None = None

        if self.seq_col is None:
            self._feature_cols = [
                c for c in train.columns
                if c != self.label_col and c not in _EXCLUDE_COLS
                and pd.api.types.is_numeric_dtype(train[c])
            ]

    def get_features(self, data: pd.DataFrame) -> np.ndarray | pd.Series:
        if self.seq_col:
            return data[self.seq_col]
        assert self._feature_cols is not None
        return data[self._feature_cols].to_numpy(dtype=np.float32)

    def get_labels(self, data: pd.DataFrame) -> np.ndarray:
        return data[self.label_col].to_numpy()

    def format_predictions(self, data_template: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
        out = data_template.copy()
        out[self.label_col] = predictions
        return out

    def profile(self, train: pd.DataFrame, test: pd.DataFrame) -> DataProfile:
        labels = self.get_labels(train)
        task_type = _infer_task_type(labels)

        if self.seq_col:
            data_type = DataType.DATAFRAME_SEQUENCE
        else:
            data_type = DataType.DATAFRAME_TABULAR

        n_features = len(self._feature_cols) if self._feature_cols else None
        n_classes = int(len(np.unique(labels))) if task_type == TaskType.CLASSIFICATION else None
        class_dist = None
        label_stats = None

        if task_type == TaskType.CLASSIFICATION:
            unique, counts = np.unique(labels, return_counts=True)
            class_dist = {str(k): int(v) for k, v in zip(unique, counts)}
        else:
            label_stats = {
                "mean": float(np.nanmean(labels)),
                "std": float(np.nanstd(labels)),
                "min": float(np.nanmin(labels)),
                "max": float(np.nanmax(labels)),
            }

        extra = {}
        if self.seq_col:
            lengths = train[self.seq_col].str.len()
            extra["seq_len_min"] = int(lengths.min())
            extra["seq_len_max"] = int(lengths.max())
            extra["seq_len_mean"] = float(lengths.mean())

        return DataProfile(
            data_type=data_type,
            task_type=task_type,
            train_samples=len(train),
            test_samples=len(test),
            n_features=n_features,
            n_classes=n_classes,
            label_column=self.label_col,
            feature_columns=self._feature_cols or [],
            sequence_column=self.seq_col,
            class_distribution=class_dist,
            label_stats=label_stats,
            extra=extra,
        )

    def save_train_data(self, train: pd.DataFrame, path: Path) -> None:
        train.to_pickle(path)

    def save_test_features(self, test: pd.DataFrame, path: Path) -> None:
        cols = [c for c in test.columns if c != self.label_col]
        test[cols].to_pickle(path)
