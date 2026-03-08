"""K-mer featurization + regression models for DNA/RNA sequence tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from genbio.coscientist.config import DataType, TaskType
from genbio.coscientist.pipelines.base import PipelineBase, PipelineRegistry


class DensifyTransformer:
    """Convert sparse matrices to dense for estimators that require it."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import scipy.sparse as sp
        return X.toarray() if sp.issparse(X) else X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def _gc_content(sequences):
    """Compute GC content as a single feature column."""
    vals = []
    for seq in sequences:
        s = seq.upper()
        gc = (s.count("G") + s.count("C")) / max(len(s), 1)
        vals.append([gc])
    return np.array(vals, dtype=np.float32)


def _seq_length(sequences):
    return np.array([[len(s)] for s in sequences], dtype=np.float32)


class KmerRidgePipeline(PipelineBase):
    name = "kmer_ridge"
    supported_data_types = [DataType.DATAFRAME_SEQUENCE]
    supported_task_types = [TaskType.REGRESSION]

    def _build_pipeline(self, config: dict) -> Pipeline:
        ngram_lo = config.get("ngram_lo", 3)
        ngram_hi = config.get("ngram_hi", 6)
        alpha = config.get("alpha", 1.0)
        use_engineered = config.get("use_engineered", True)

        transformers = [
            ("kmer", CountVectorizer(analyzer="char", ngram_range=(ngram_lo, ngram_hi))),
        ]
        if use_engineered:
            transformers.extend([
                ("gc", FunctionTransformer(_gc_content, validate=False)),
                ("seqlen", FunctionTransformer(_seq_length, validate=False)),
            ])

        return Pipeline([
            ("features", FeatureUnion(transformers)),
            ("scale", StandardScaler(with_mean=False)),
            ("model", Ridge(alpha=alpha)),
        ])

    def train(self, X: Any, y: np.ndarray, config: dict | None = None) -> Pipeline:
        config = config or self.get_default_config()
        pipe = self._build_pipeline(config)
        pipe.fit(X, y)
        return pipe

    def predict(self, model: Pipeline, X: Any) -> np.ndarray:
        return model.predict(X)

    def get_default_config(self) -> dict:
        return {"ngram_lo": 3, "ngram_hi": 6, "alpha": 1.0, "use_engineered": True}

    def get_search_space(self, trial) -> dict:
        return {
            "ngram_lo": trial.suggest_int("ngram_lo", 2, 4),
            "ngram_hi": trial.suggest_int("ngram_hi", 5, 8),
            "alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True),
            "use_engineered": trial.suggest_categorical("use_engineered", [True, False]),
        }


class KmerElasticNetPipeline(PipelineBase):
    name = "kmer_elasticnet"
    supported_data_types = [DataType.DATAFRAME_SEQUENCE]
    supported_task_types = [TaskType.REGRESSION]

    def _build_pipeline(self, config: dict) -> Pipeline:
        ngram_lo = config.get("ngram_lo", 3)
        ngram_hi = config.get("ngram_hi", 6)

        transformers = [
            ("kmer", CountVectorizer(analyzer="char", ngram_range=(ngram_lo, ngram_hi))),
            ("gc", FunctionTransformer(_gc_content, validate=False)),
            ("seqlen", FunctionTransformer(_seq_length, validate=False)),
        ]

        return Pipeline([
            ("features", FeatureUnion(transformers)),
            ("scale", StandardScaler(with_mean=False)),
            ("model", ElasticNet(
                alpha=config.get("alpha", 0.1),
                l1_ratio=config.get("l1_ratio", 0.5),
                max_iter=5000,
            )),
        ])

    def train(self, X: Any, y: np.ndarray, config: dict | None = None) -> Pipeline:
        config = config or self.get_default_config()
        pipe = self._build_pipeline(config)
        pipe.fit(X, y)
        return pipe

    def predict(self, model: Pipeline, X: Any) -> np.ndarray:
        return model.predict(X)

    def get_default_config(self) -> dict:
        return {"ngram_lo": 3, "ngram_hi": 6, "alpha": 0.1, "l1_ratio": 0.5}

    def get_search_space(self, trial) -> dict:
        return {
            "ngram_lo": trial.suggest_int("ngram_lo", 2, 4),
            "ngram_hi": trial.suggest_int("ngram_hi", 5, 8),
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
        }


class KmerGBRPipeline(PipelineBase):
    name = "kmer_hgbr"
    supported_data_types = [DataType.DATAFRAME_SEQUENCE]
    supported_task_types = [TaskType.REGRESSION]

    def _build_pipeline(self, config: dict) -> Pipeline:
        ngram_lo = config.get("ngram_lo", 3)
        ngram_hi = config.get("ngram_hi", 6)

        transformers = [
            ("kmer", CountVectorizer(analyzer="char", ngram_range=(ngram_lo, ngram_hi))),
            ("gc", FunctionTransformer(_gc_content, validate=False)),
            ("seqlen", FunctionTransformer(_seq_length, validate=False)),
        ]

        return Pipeline([
            ("features", FeatureUnion(transformers)),
            ("densify", DensifyTransformer()),
            ("model", HistGradientBoostingRegressor(
                max_iter=config.get("max_iter", 200),
                learning_rate=config.get("learning_rate", 0.1),
                max_depth=config.get("max_depth", 6),
                min_samples_leaf=config.get("min_samples_leaf", 10),
                l2_regularization=config.get("l2_regularization", 0.1),
                random_state=config.get("seed", 42),
            )),
        ])

    def train(self, X: Any, y: np.ndarray, config: dict | None = None) -> Pipeline:
        config = config or self.get_default_config()
        pipe = self._build_pipeline(config)
        pipe.fit(X, y)
        return pipe

    def predict(self, model: Pipeline, X: Any) -> np.ndarray:
        return model.predict(X)

    def get_default_config(self) -> dict:
        return {
            "ngram_lo": 3, "ngram_hi": 6,
            "max_iter": 200, "learning_rate": 0.1,
            "max_depth": 6, "min_samples_leaf": 10,
            "l2_regularization": 0.1, "seed": 42,
        }

    def get_search_space(self, trial) -> dict:
        return {
            "ngram_lo": trial.suggest_int("ngram_lo", 2, 4),
            "ngram_hi": trial.suggest_int("ngram_hi", 5, 8),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        }


PipelineRegistry.register(KmerRidgePipeline())
PipelineRegistry.register(KmerElasticNetPipeline())
PipelineRegistry.register(KmerGBRPipeline())
