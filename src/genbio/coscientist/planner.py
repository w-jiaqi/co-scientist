"""Heuristic experiment planner: selects pipelines and configs based on data profile."""

from __future__ import annotations

from dataclasses import dataclass, field

from genbio.coscientist.config import BudgetTier, DataProfile, DataType, TaskType
from genbio.coscientist.pipelines.base import PipelineBase, PipelineRegistry

# Ensure all pipelines are registered
import genbio.coscientist.pipelines.kmer_regression  # noqa: F401
import genbio.coscientist.pipelines.expression_classification  # noqa: F401
import genbio.coscientist.pipelines.generic_tabular  # noqa: F401


@dataclass
class ExperimentPlan:
    experiments: list[PlannedExperiment] = field(default_factory=list)
    do_ensemble: bool = False
    reasoning: str = ""

    def summary(self) -> str:
        lines = [self.reasoning, "", "Planned experiments:"]
        for i, exp in enumerate(self.experiments):
            lines.append(f"  {i+1}. {exp.name} ({exp.pipeline.name})")
        if self.do_ensemble:
            lines.append("  + Ensemble of top models")
        return "\n".join(lines)


@dataclass
class PlannedExperiment:
    name: str
    pipeline: PipelineBase
    config: dict
    tune: bool = False
    optuna_trials: int = 0


def create_plan(profile: DataProfile, budget: BudgetTier) -> ExperimentPlan:
    """Build an experiment plan based on dataset characteristics and budget."""
    candidates = PipelineRegistry.get_candidates(profile.data_type, profile.task_type)

    if not candidates:
        candidates = PipelineRegistry.get_candidates(DataType.DATAFRAME_TABULAR, profile.task_type)

    if not candidates:
        raise ValueError(
            f"No pipelines available for data_type={profile.data_type}, task_type={profile.task_type}. "
            f"Available pipelines: {[p.name for p in PipelineRegistry.all_pipelines()]}"
        )

    max_models = budget.max_models
    trials_per_model = budget.optuna_trials

    experiments: list[PlannedExperiment] = []

    first = candidates[0]
    experiments.append(PlannedExperiment(
        name=f"baseline_{first.name}",
        pipeline=first,
        config=first.get_default_config(),
        tune=False,
    ))

    for pipe in candidates[:max_models]:
        experiments.append(PlannedExperiment(
            name=f"tuned_{pipe.name}",
            pipeline=pipe,
            config=pipe.get_default_config(),
            tune=True,
            optuna_trials=trials_per_model,
        ))

    reasoning = _build_reasoning(profile, candidates[:max_models], budget)

    return ExperimentPlan(
        experiments=experiments,
        do_ensemble=budget.do_ensemble,
        reasoning=reasoning,
    )


def _build_reasoning(profile: DataProfile, pipelines: list[PipelineBase], budget: BudgetTier) -> str:
    lines: list[str] = []

    if profile.data_type == DataType.DATAFRAME_SEQUENCE:
        lines.append(
            f"Detected DNA/RNA sequence data ({profile.train_samples} train samples, "
            f"sequences ~{profile.extra.get('seq_len_mean', '?'):.0f}bp)."
        )
        lines.append("Using k-mer featurization with character n-grams.")
        lines.append(f"Task: {profile.task_type.value} — primary metric drives model selection.")
    elif profile.data_type == DataType.ANNDATA:
        lines.append(
            f"Detected single-cell expression data ({profile.train_samples} cells, "
            f"{profile.n_features} genes, {profile.n_classes} classes)."
        )
        lines.append("Preprocessing: normalize → log1p → PCA; using balanced classifiers for macro F1.")
    else:
        lines.append(
            f"Detected tabular data ({profile.train_samples} samples, "
            f"{profile.n_features} features)."
        )

    lines.append(f"Budget: {budget.value} — {len(pipelines)} models, {budget.optuna_trials} Optuna trials each.")
    if budget.do_ensemble:
        lines.append("Will ensemble top performers.")

    return " ".join(lines)
