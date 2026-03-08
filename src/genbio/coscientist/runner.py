"""Experiment execution engine: trains, evaluates, and selects models."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from genbio.coscientist.config import (
    BudgetTier,
    DataProfile,
    DataType,
    ExperimentResult,
    RunConfig,
    TaskType,
)
from genbio.coscientist.adapters.base import DatasetAdapter
from genbio.coscientist.planner import ExperimentPlan, PlannedExperiment
from genbio.coscientist import console as ui


def _get_cv_splitter(profile: DataProfile, train_data: Any, adapter: DatasetAdapter, seed: int):
    """Create a cross-validation iterator appropriate for the dataset."""
    from sklearn.model_selection import StratifiedKFold, KFold
    import pandas as pd

    y = adapter.get_labels(train_data)

    if isinstance(train_data, pd.DataFrame) and "fold_id" in train_data.columns:
        folds = train_data["fold_id"].values
        unique_folds = np.unique(folds)
        if len(unique_folds) >= 3:
            # Use existing fold structure for inner CV (pick 2 held-out folds)
            indices = []
            for val_fold in unique_folds[:3]:
                train_idx = np.where(folds != val_fold)[0]
                val_idx = np.where(folds == val_fold)[0]
                indices.append((train_idx, val_idx))
            return indices

    if profile.task_type == TaskType.CLASSIFICATION:
        unique, counts = np.unique(y, return_counts=True)
        min_count = int(counts.min())
        n_splits = min(3, min_count) if min_count >= 2 else 2
        if n_splits < 2:
            # Fallback: simple random split
            from sklearn.model_selection import ShuffleSplit
            return ShuffleSplit(n_splits=2, test_size=0.25, random_state=seed)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=3, shuffle=True, random_state=seed)


def _evaluate_cv(
    pipeline, X, y, cv, profile: DataProfile, config: dict
) -> float:
    """Run cross-validation and return the mean primary metric."""
    from scipy.stats import spearmanr
    from sklearn.metrics import f1_score

    scores = []
    splits = cv if isinstance(cv, list) else cv.split(X, y)

    for train_idx, val_idx in splits:
        if isinstance(X, np.ndarray):
            X_train, X_val = X[train_idx], X[val_idx]
        else:
            X_train = X.iloc[train_idx] if hasattr(X, "iloc") else [X[i] for i in train_idx]
            X_val = X.iloc[val_idx] if hasattr(X, "iloc") else [X[i] for i in val_idx]

        y_train, y_val = y[train_idx], y[val_idx]

        model = pipeline.train(X_train, y_train, config)
        preds = pipeline.predict(model, X_val)

        if profile.task_type == TaskType.REGRESSION:
            corr, _ = spearmanr(preds, y_val)
            scores.append(corr if not np.isnan(corr) else 0.0)
        else:
            scores.append(f1_score(y_val, preds, average="macro", zero_division=0))

    return float(np.mean(scores))


class ExperimentRunner:
    """Runs experiments, tunes hyperparameters, and manages artifacts."""

    def __init__(self, config: RunConfig, adapter: DatasetAdapter, profile: DataProfile):
        self.config = config
        self.adapter = adapter
        self.profile = profile
        self.results: list[ExperimentResult] = []

    def execute(self, plan: ExperimentPlan, train_data: Any, test_data: Any) -> list[ExperimentResult]:
        """Execute all planned experiments."""
        X = self.adapter.get_features(train_data)
        y = self.adapter.get_labels(train_data)
        cv = _get_cv_splitter(self.profile, train_data, self.adapter, self.config.seed)

        primary_metric_name = self._detect_primary_metric()

        self.results = []

        for exp in plan.experiments:
            ui.step(len(self.results) + 1, f"Running: [cyan]{exp.name}[/cyan]")
            t0 = time.time()

            if exp.tune and exp.optuna_trials > 0:
                best_config, best_score = self._tune(exp, X, y, cv)
            else:
                best_config = exp.config
                best_score = _evaluate_cv(exp.pipeline, X, y, cv, self.profile, best_config)

            duration = time.time() - t0

            # Train final model on full training data
            model = exp.pipeline.train(X, y, best_config)
            model_dir = self.config.run_dir / "experiments" / exp.name
            model_path = model_dir / "model.joblib"
            exp.pipeline.save_model(model, model_path)

            (model_dir / "config.json").write_text(json.dumps(best_config, indent=2))
            metrics_dict = {primary_metric_name: best_score}
            (model_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))

            result = ExperimentResult(
                name=exp.name,
                pipeline_name=exp.pipeline.name,
                config=best_config,
                metrics=metrics_dict,
                primary_metric_name=primary_metric_name,
                primary_metric_value=best_score,
                model_path=model_path,
                duration_seconds=duration,
            )
            self.results.append(result)
            ui.success(f"{exp.name}: {primary_metric_name} = {best_score:.4f} ({duration:.1f}s)")

        if plan.do_ensemble and len(self.results) >= 2:
            self._run_ensemble(X, y, cv, plan, train_data, test_data, primary_metric_name)

        return self.results

    def _tune(self, exp: PlannedExperiment, X, y, cv) -> tuple[dict, float]:
        """Hyperparameter tuning with Optuna."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            config = exp.pipeline.get_search_space(trial)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return _evaluate_cv(exp.pipeline, X, y, cv, self.profile, config)

        direction = "maximize"
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=exp.optuna_trials, show_progress_bar=False)

        return study.best_params, study.best_value

    def _run_ensemble(self, X, y, cv, plan, train_data, test_data, primary_metric_name):
        """Simple averaging ensemble of top models."""
        sorted_results = sorted(self.results, key=lambda r: r.primary_metric_value, reverse=True)
        top_two = sorted_results[:2]

        ui.step(len(self.results) + 1, f"Running: [cyan]ensemble ({top_two[0].name} + {top_two[1].name})[/cyan]")
        t0 = time.time()

        # Load both models
        models = []
        pipelines = []
        for r in top_two:
            pipe = self._find_pipeline(r.pipeline_name, plan)
            model = pipe.load_model(r.model_path)
            models.append(model)
            pipelines.append(pipe)

        # CV evaluate the ensemble
        from scipy.stats import spearmanr
        from sklearn.metrics import f1_score

        scores = []
        splits = cv if isinstance(cv, list) else list(cv.split(X, y))

        for train_idx, val_idx in splits:
            if isinstance(X, np.ndarray):
                X_train, X_val = X[train_idx], X[val_idx]
            else:
                X_train = X.iloc[train_idx] if hasattr(X, "iloc") else [X[i] for i in train_idx]
                X_val = X.iloc[val_idx] if hasattr(X, "iloc") else [X[i] for i in val_idx]

            y_train, y_val = y[train_idx], y[val_idx]

            preds_list = []
            for pipe_obj, r in zip(pipelines, top_two):
                m = pipe_obj.train(X_train, y_train, r.config)
                preds_list.append(pipe_obj.predict(m, X_val))

            if self.profile.task_type == TaskType.REGRESSION:
                ensemble_preds = np.mean(preds_list, axis=0)
                corr, _ = spearmanr(ensemble_preds, y_val)
                scores.append(corr if not np.isnan(corr) else 0.0)
            else:
                # Majority vote for classification
                stacked = np.array(preds_list)
                from scipy.stats import mode as scipy_mode
                ensemble_preds = scipy_mode(stacked, axis=0, keepdims=False).mode
                scores.append(f1_score(y_val, ensemble_preds, average="macro", zero_division=0))

        ensemble_score = float(np.mean(scores))
        duration = time.time() - t0

        model_dir = self.config.run_dir / "experiments" / "ensemble"
        model_dir.mkdir(parents=True, exist_ok=True)
        ensemble_info = {
            "components": [r.name for r in top_two],
            "component_paths": [str(r.model_path) for r in top_two],
        }
        (model_dir / "config.json").write_text(json.dumps(ensemble_info, indent=2))
        (model_dir / "metrics.json").write_text(json.dumps({primary_metric_name: ensemble_score}, indent=2))

        result = ExperimentResult(
            name="ensemble",
            pipeline_name="ensemble",
            config=ensemble_info,
            metrics={primary_metric_name: ensemble_score},
            primary_metric_name=primary_metric_name,
            primary_metric_value=ensemble_score,
            model_path=top_two[0].model_path,
            duration_seconds=duration,
        )
        self.results.append(result)
        ui.success(f"ensemble: {primary_metric_name} = {ensemble_score:.4f} ({duration:.1f}s)")

    def _find_pipeline(self, name: str, plan: ExperimentPlan):
        for exp in plan.experiments:
            if exp.pipeline.name == name:
                return exp.pipeline
        from genbio.coscientist.pipelines.base import PipelineRegistry
        for p in PipelineRegistry.all_pipelines():
            if p.name == name:
                return p
        raise ValueError(f"Pipeline {name} not found")

    def select_best(self) -> ExperimentResult:
        """Return the experiment with the highest primary metric."""
        if not self.results:
            raise RuntimeError("No experiments have been run yet.")
        return max(self.results, key=lambda r: r.primary_metric_value)

    def _detect_primary_metric(self) -> str:
        if self.profile.task_type == TaskType.REGRESSION:
            return "spearman"
        return "f1_macro"

    def generate_test_predictions(
        self, best: ExperimentResult, plan: ExperimentPlan,
        train_data: Any, test_data: Any,
    ) -> Any:
        """Generate predictions on test data using the best model."""
        X_test = self.adapter.get_features(test_data)

        if best.pipeline_name == "ensemble":
            component_paths = best.config.get("component_paths", [])
            preds_list = []
            for path_str in component_paths:
                path = Path(path_str)
                pipeline_name_parts = path.parent.name.split("_", 1)
                # Try to find the pipeline from the result that produced this model
                for r in self.results:
                    if str(r.model_path) == path_str:
                        pipe = self._find_pipeline(r.pipeline_name, plan)
                        model = pipe.load_model(path)
                        preds_list.append(pipe.predict(model, X_test))
                        break

            if self.profile.task_type == TaskType.REGRESSION:
                raw_preds = np.mean(preds_list, axis=0)
            else:
                from scipy.stats import mode as scipy_mode
                stacked = np.array(preds_list)
                raw_preds = scipy_mode(stacked, axis=0, keepdims=False).mode
        else:
            pipe = self._find_pipeline(best.pipeline_name, plan)
            model = pipe.load_model(best.model_path)
            raw_preds = pipe.predict(model, X_test)

        return self.adapter.format_predictions(test_data, raw_preds)
