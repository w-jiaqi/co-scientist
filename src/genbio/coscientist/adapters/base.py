"""Abstract base class for dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from genbio.coscientist.config import DataProfile


class DatasetAdapter(ABC):
    """Normalises different data formats into a uniform interface."""

    @abstractmethod
    def get_features(self, data: Any) -> np.ndarray:
        """Extract feature matrix from dataset."""
        ...

    @abstractmethod
    def get_labels(self, data: Any) -> np.ndarray:
        """Extract label vector from dataset."""
        ...

    @abstractmethod
    def format_predictions(self, data_template: Any, predictions: np.ndarray) -> Any:
        """Wrap raw prediction array back into the format expected by evaluate()."""
        ...

    @abstractmethod
    def profile(self, train: Any, test: Any) -> DataProfile:
        """Generate a DataProfile describing the dataset."""
        ...

    @abstractmethod
    def save_train_data(self, train: Any, path: Any) -> None:
        """Persist training data for agent workspace or caching."""
        ...

    @abstractmethod
    def save_test_features(self, test: Any, path: Any) -> None:
        """Persist test features (no labels) for agent workspace."""
        ...


def get_adapter(train: Any, test: Any) -> DatasetAdapter:
    """Auto-detect and return the appropriate adapter."""
    import pandas as pd

    try:
        import anndata
        if isinstance(train, anndata.AnnData):
            from genbio.coscientist.adapters.anndata_adapter import AnnDataAdapter
            return AnnDataAdapter(train, test)
    except ImportError:
        pass

    if isinstance(train, pd.DataFrame):
        from genbio.coscientist.adapters.dataframe_adapter import DataFrameAdapter
        return DataFrameAdapter(train, test)

    raise TypeError(f"Unsupported data type: {type(train).__name__}")
