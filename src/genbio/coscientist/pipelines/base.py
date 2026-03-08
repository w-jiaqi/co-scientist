"""Abstract pipeline interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from genbio.coscientist.config import DataType, TaskType


class PipelineBase(ABC):
    """A self-contained ML pipeline: preprocessing + model."""

    name: str
    supported_data_types: list[DataType]
    supported_task_types: list[TaskType]

    @abstractmethod
    def train(self, X: Any, y: np.ndarray, config: dict | None = None) -> Any:
        """Train and return the fitted pipeline/model."""
        ...

    @abstractmethod
    def predict(self, model: Any, X: Any) -> np.ndarray:
        """Generate predictions from a fitted model."""
        ...

    @abstractmethod
    def get_search_space(self, trial) -> dict:
        """Return Optuna-compatible hyperparameter suggestions."""
        ...

    @abstractmethod
    def get_default_config(self) -> dict:
        """Return sensible defaults (no tuning)."""
        ...

    def save_model(self, model: Any, path) -> None:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)

    def load_model(self, path) -> Any:
        import joblib
        return joblib.load(path)


class PipelineRegistry:
    """Maps (data_type, task_type) to candidate pipelines."""

    _pipelines: list[PipelineBase] = []

    @classmethod
    def register(cls, pipeline: PipelineBase) -> PipelineBase:
        cls._pipelines.append(pipeline)
        return pipeline

    @classmethod
    def get_candidates(cls, data_type: DataType, task_type: TaskType) -> list[PipelineBase]:
        return [
            p for p in cls._pipelines
            if data_type in p.supported_data_types and task_type in p.supported_task_types
        ]

    @classmethod
    def all_pipelines(cls) -> list[PipelineBase]:
        return list(cls._pipelines)
