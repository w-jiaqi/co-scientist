"""Adapter for AnnData (single-cell expression) datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from genbio.coscientist.adapters.base import DatasetAdapter
from genbio.coscientist.config import DataProfile, DataType, TaskType

_LABEL_CANDIDATES = ["cell_type_label", "cell_type", "label", "labels", "celltype"]


def _detect_label_col(adata) -> str:
    for c in _LABEL_CANDIDATES:
        if c in adata.obs.columns:
            return c
    label_cols = [c for c in adata.obs.columns if "label" in c.lower()]
    if label_cols:
        return label_cols[0]
    cat_cols = [c for c in adata.obs.columns if adata.obs[c].dtype.name == "category"
                or adata.obs[c].nunique() < 50]
    if len(cat_cols) == 1:
        return cat_cols[0]
    raise ValueError(
        f"Cannot auto-detect label column in obs. Columns: {list(adata.obs.columns)}. "
        f"Expected one of {_LABEL_CANDIDATES}."
    )


class AnnDataAdapter(DatasetAdapter):
    def __init__(self, train, test) -> None:
        self.label_col = _detect_label_col(train)

    def get_features(self, data) -> np.ndarray:
        import scipy.sparse as sp
        X = data.X
        if sp.issparse(X):
            return np.asarray(X.toarray(), dtype=np.float32)
        return np.asarray(X, dtype=np.float32)

    def get_labels(self, data) -> np.ndarray:
        return data.obs[self.label_col].values

    def format_predictions(self, data_template, predictions: np.ndarray):
        out = data_template.copy()
        out.obs[self.label_col] = predictions
        return out

    def profile(self, train, test) -> DataProfile:
        labels = self.get_labels(train)
        unique, counts = np.unique(labels, return_counts=True)

        return DataProfile(
            data_type=DataType.ANNDATA,
            task_type=TaskType.CLASSIFICATION,
            train_samples=train.shape[0],
            test_samples=test.shape[0],
            n_features=train.shape[1],
            n_classes=int(len(unique)),
            label_column=self.label_col,
            class_distribution={str(k): int(v) for k, v in zip(unique, counts)},
            extra={"n_genes": train.shape[1]},
        )

    def save_train_data(self, train, path: Path) -> None:
        train.write_h5ad(path.with_suffix(".h5ad"))

    def save_test_features(self, test, path: Path) -> None:
        import anndata
        out = anndata.AnnData(X=test.X, var=test.var)
        out.write_h5ad(path.with_suffix(".h5ad"))
