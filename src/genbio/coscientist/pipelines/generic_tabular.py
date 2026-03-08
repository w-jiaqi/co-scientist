"""Generic tabular pipeline fallback for unknown DataFrame datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from genbio.coscientist.config import DataType, TaskType
from genbio.coscientist.pipelines.base import PipelineBase, PipelineRegistry


class TabularRidgePipeline(PipelineBase):
    name = "tabular_ridge"
    supported_data_types = [DataType.DATAFRAME_TABULAR]
    supported_task_types = [TaskType.REGRESSION]

    def train(self, X: np.ndarray, y: np.ndarray, config: dict | None = None) -> Pipeline:
        config = config or self.get_default_config()
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=config.get("alpha", 1.0))),
        ])
        pipe.fit(X, y)
        return pipe

    def predict(self, model: Pipeline, X: np.ndarray) -> np.ndarray:
        return model.predict(X)

    def get_default_config(self) -> dict:
        return {"alpha": 1.0}

    def get_search_space(self, trial) -> dict:
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}


class TabularGBRPipeline(PipelineBase):
    name = "tabular_hgbr"
    supported_data_types = [DataType.DATAFRAME_TABULAR]
    supported_task_types = [TaskType.REGRESSION]

    def train(self, X: np.ndarray, y: np.ndarray, config: dict | None = None) -> Pipeline:
        config = config or self.get_default_config()
        pipe = Pipeline([
            ("model", HistGradientBoostingRegressor(
                max_iter=config.get("max_iter", 200),
                learning_rate=config.get("learning_rate", 0.1),
                max_depth=config.get("max_depth", 6),
                random_state=42,
            )),
        ])
        pipe.fit(X, y)
        return pipe

    def predict(self, model: Pipeline, X: np.ndarray) -> np.ndarray:
        return model.predict(X)

    def get_default_config(self) -> dict:
        return {"max_iter": 200, "learning_rate": 0.1, "max_depth": 6}

    def get_search_space(self, trial) -> dict:
        return {
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
        }


class TabularLogRegPipeline(PipelineBase):
    name = "tabular_logreg"
    supported_data_types = [DataType.DATAFRAME_TABULAR]
    supported_task_types = [TaskType.CLASSIFICATION]

    def train(self, X: np.ndarray, y: np.ndarray, config: dict | None = None) -> Pipeline:
        config = config or self.get_default_config()
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=config.get("C", 1.0),
                max_iter=2000,
                solver="saga",
                class_weight="balanced",
                random_state=42,
            )),
        ])
        pipe.fit(X, y)
        return pipe

    def predict(self, model: Pipeline, X: np.ndarray) -> np.ndarray:
        return model.predict(X)

    def get_default_config(self) -> dict:
        return {"C": 1.0}

    def get_search_space(self, trial) -> dict:
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}


class TabularGBCPipeline(PipelineBase):
    name = "tabular_hgbc"
    supported_data_types = [DataType.DATAFRAME_TABULAR]
    supported_task_types = [TaskType.CLASSIFICATION]

    def train(self, X: np.ndarray, y: np.ndarray, config: dict | None = None) -> Pipeline:
        config = config or self.get_default_config()
        pipe = Pipeline([
            ("model", HistGradientBoostingClassifier(
                max_iter=config.get("max_iter", 200),
                learning_rate=config.get("learning_rate", 0.1),
                max_depth=config.get("max_depth", 6),
                class_weight="balanced",
                random_state=42,
            )),
        ])
        pipe.fit(X, y)
        return pipe

    def predict(self, model: Pipeline, X: np.ndarray) -> np.ndarray:
        return model.predict(X)

    def get_default_config(self) -> dict:
        return {"max_iter": 200, "learning_rate": 0.1, "max_depth": 6}

    def get_search_space(self, trial) -> dict:
        return {
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
        }


class TabularRFPipeline(PipelineBase):
    name = "tabular_rf"
    supported_data_types = [DataType.DATAFRAME_TABULAR]
    supported_task_types = [TaskType.CLASSIFICATION]

    def train(self, X: np.ndarray, y: np.ndarray, config: dict | None = None) -> Pipeline:
        config = config or self.get_default_config()
        pipe = Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=config.get("n_estimators", 200),
                max_depth=config.get("max_depth", None),
                class_weight="balanced",
                random_state=42,
            )),
        ])
        pipe.fit(X, y)
        return pipe

    def predict(self, model: Pipeline, X: np.ndarray) -> np.ndarray:
        return model.predict(X)

    def get_default_config(self) -> dict:
        return {"n_estimators": 200, "max_depth": None}

    def get_search_space(self, trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
        }


PipelineRegistry.register(TabularRidgePipeline())
PipelineRegistry.register(TabularGBRPipeline())
PipelineRegistry.register(TabularLogRegPipeline())
PipelineRegistry.register(TabularGBCPipeline())
PipelineRegistry.register(TabularRFPipeline())
