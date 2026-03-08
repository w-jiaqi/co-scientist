"""Dataset analysis and profiling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from genbio.coscientist.adapters import get_adapter, DatasetAdapter
from genbio.coscientist.config import DataProfile


def discover_datasets() -> list[str]:
    """Scan genbio.datasets subpackages and return available dataset names."""
    datasets_dir = Path(__file__).parent.parent / "datasets"
    results: list[str] = []

    for category_dir in sorted(datasets_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("_"):
            continue
        for task_dir in sorted(category_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name.startswith("_"):
                continue
            load_py = task_dir / "load.py"
            if load_py.exists():
                name = f"{category_dir.name}/{task_dir.name}".replace("_", "-")
                results.append(name)

    return results


def analyze_dataset(train: Any, test: Any) -> tuple[DatasetAdapter, DataProfile]:
    """Auto-detect adapter and profile the dataset."""
    adapter = get_adapter(train, test)
    profile = adapter.profile(train, test)
    return adapter, profile
