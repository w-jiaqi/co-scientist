"""Expression data classification pipelines (scanpy preprocessing + classifiers)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from genbio.coscientist.config import DataType, TaskType
from genbio.coscientist.pipelines.base import PipelineBase, PipelineRegistry


def _scanpy_preprocess(X: np.ndarray, config: dict) -> np.ndarray:
    """Mimic scanpy normalize_total + log1p + HVG + PCA without requiring scanpy at inference."""
    from sklearn.preprocessing import normalize

    X_norm = normalize(X, norm="l1", axis=1) * config.get("target_sum", 1e4)
    X_log = np.log1p(X_norm)
    return X_log


class ExprLogRegPipeline(PipelineBase):
    name = "expr_logreg"
    supported_data_types = [DataType.ANNDATA]
    supported_task_types = [TaskType.CLASSIFICATION]

    def train(self, X: np.ndarray, y: np.ndarray, config: dict | None = None) -> Pipeline:
        config = config or self.get_default_config()
        X_proc = _scanpy_preprocess(X, config)

        pipe = Pipeline([
            ("scale", StandardScaler(with_mean=True)),
            ("pca", PCA(n_components=config.get("n_components", 50), random_state=42)),
            ("model", LogisticRegression(
                C=config.get("C", 1.0),
                max_iter=config.get("max_iter", 2000),
                solver="saga",
                class_weight="balanced",
                random_state=42,
            )),
        ])
        pipe.fit(X_proc, y)
        pipe._cosci_preprocess_config = config
        return pipe

    def predict(self, model: Pipeline, X: np.ndarray) -> np.ndarray:
        config = getattr(model, "_cosci_preprocess_config", {})
        X_proc = _scanpy_preprocess(X, config)
        return model.predict(X_proc)

    def get_default_config(self) -> dict:
        return {"n_components": 50, "C": 1.0, "max_iter": 2000, "target_sum": 1e4}

    def get_search_space(self, trial) -> dict:
        return {
            "n_components": trial.suggest_int("n_components", 20, 200),
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "max_iter": 2000,
            "target_sum": 1e4,
        }


class ExprSVMPipeline(PipelineBase):
    name = "expr_svm"
    supported_data_types = [DataType.ANNDATA]
    supported_task_types = [TaskType.CLASSIFICATION]

    def train(self, X: np.ndarray, y: np.ndarray, config: dict | None = None) -> Pipeline:
        config = config or self.get_default_config()
        X_proc = _scanpy_preprocess(X, config)

        base_svm = LinearSVC(
            C=config.get("C", 1.0),
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )
        unique, counts = np.unique(y, return_counts=True)
        cv_folds = min(3, int(counts.min())) if int(counts.min()) >= 2 else 2
        pipe = Pipeline([
            ("scale", StandardScaler(with_mean=True)),
            ("pca", PCA(n_components=config.get("n_components", 50), random_state=42)),
            ("model", CalibratedClassifierCV(base_svm, cv=cv_folds)),
        ])
        pipe.fit(X_proc, y)
        pipe._cosci_preprocess_config = config
        return pipe

    def predict(self, model: Pipeline, X: np.ndarray) -> np.ndarray:
        config = getattr(model, "_cosci_preprocess_config", {})
        X_proc = _scanpy_preprocess(X, config)
        return model.predict(X_proc)

    def get_default_config(self) -> dict:
        return {"n_components": 50, "C": 1.0, "target_sum": 1e4}

    def get_search_space(self, trial) -> dict:
        return {
            "n_components": trial.suggest_int("n_components", 20, 200),
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "target_sum": 1e4,
        }


class ExprMLPPipeline(PipelineBase):
    name = "expr_mlp"
    supported_data_types = [DataType.ANNDATA]
    supported_task_types = [TaskType.CLASSIFICATION]

    def train(self, X: np.ndarray, y: np.ndarray, config: dict | None = None) -> Pipeline:
        config = config or self.get_default_config()
        X_proc = _scanpy_preprocess(X, config)

        h1 = config.get("hidden1", 256)
        h2 = config.get("hidden2", 128)
        pipe = Pipeline([
            ("scale", StandardScaler(with_mean=True)),
            ("pca", PCA(n_components=config.get("n_components", 50), random_state=42)),
            ("model", MLPClassifier(
                hidden_layer_sizes=(h1, h2),
                activation="relu",
                alpha=config.get("alpha", 1e-3),
                learning_rate_init=config.get("lr", 1e-3),
                max_iter=config.get("max_iter", 500),
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            )),
        ])
        pipe.fit(X_proc, y)
        pipe._cosci_preprocess_config = config
        return pipe

    def predict(self, model: Pipeline, X: np.ndarray) -> np.ndarray:
        config = getattr(model, "_cosci_preprocess_config", {})
        X_proc = _scanpy_preprocess(X, config)
        return model.predict(X_proc)

    def get_default_config(self) -> dict:
        return {
            "n_components": 50, "hidden1": 256, "hidden2": 128,
            "alpha": 1e-3, "lr": 1e-3, "max_iter": 500, "target_sum": 1e4,
        }

    def get_search_space(self, trial) -> dict:
        return {
            "n_components": trial.suggest_int("n_components", 20, 200),
            "hidden1": trial.suggest_categorical("hidden1", [128, 256, 512]),
            "hidden2": trial.suggest_categorical("hidden2", [64, 128, 256]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "max_iter": 500,
            "target_sum": 1e4,
        }


PipelineRegistry.register(ExprLogRegPipeline())
PipelineRegistry.register(ExprSVMPipeline())
PipelineRegistry.register(ExprMLPPipeline())
