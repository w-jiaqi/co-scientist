"""Configuration dataclasses for co-scientist runs."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml


class BudgetTier(str, Enum):
    FAST = "fast"
    STANDARD = "standard"
    THOROUGH = "thorough"

    @property
    def optuna_trials(self) -> int:
        return {self.FAST: 10, self.STANDARD: 30, self.THOROUGH: 60}[self]

    @property
    def max_models(self) -> int:
        return {self.FAST: 2, self.STANDARD: 3, self.THOROUGH: 5}[self]

    @property
    def do_ensemble(self) -> bool:
        return self in (self.STANDARD, self.THOROUGH)


class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class DataType(str, Enum):
    DATAFRAME_SEQUENCE = "dataframe_sequence"
    DATAFRAME_TABULAR = "dataframe_tabular"
    ANNDATA = "anndata"


@dataclass
class DataProfile:
    data_type: DataType
    task_type: TaskType
    train_samples: int
    test_samples: int
    n_features: int | None = None
    n_classes: int | None = None
    label_column: str = "labels"
    feature_columns: list[str] = field(default_factory=list)
    sequence_column: str | None = None
    class_distribution: dict[str, int] | None = None
    label_stats: dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_type"] = self.data_type.value
        d["task_type"] = self.task_type.value
        return d

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))


@dataclass
class ExperimentResult:
    name: str
    pipeline_name: str
    config: dict
    metrics: dict[str, float]
    primary_metric_name: str
    primary_metric_value: float
    model_path: Path | None = None
    duration_seconds: float = 0.0

    @property
    def is_baseline(self) -> bool:
        return "baseline" in self.name.lower()


@dataclass
class RunConfig:
    dataset: str
    fold: str
    user: str = "cosci"
    mode: Literal["interactive", "auto"] = "auto"
    strategy: Literal["builtin", "agentic", "hybrid"] = "builtin"
    budget: BudgetTier = BudgetTier.STANDARD
    agent: Literal["claude", "codex", "auto"] = "auto"
    seed: int = 42
    device: str = "auto"
    output_dir: Path = field(default_factory=lambda: Path("runs"))
    submit: bool = False
    run_id: str = field(default_factory=lambda: "")

    def __post_init__(self) -> None:
        if isinstance(self.budget, str):
            self.budget = BudgetTier(self.budget)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if not self.run_id:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            short = uuid.uuid4().hex[:6]
            ds_slug = self.dataset.replace("/", "_").replace("-", "_")
            self.run_id = f"{ts}_{ds_slug}_f{self.fold}_{short}"

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_id

    def save(self, path: Path | None = None) -> Path:
        p = path or (self.run_dir / "config.yaml")
        p.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["budget"] = self.budget.value
        d["output_dir"] = str(self.output_dir)
        p.write_text(yaml.dump(d, default_flow_style=False, sort_keys=False))
        return p

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        d = yaml.safe_load(path.read_text())
        return cls(**d)
