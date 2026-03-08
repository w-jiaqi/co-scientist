import yaml
import numpy as np
import pandas as pd
import importlib
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from pathlib import Path
import json


def _get_datasets_dir():
    """Get the path to the datasets directory."""
    # Assume datasets directory is at the root of the repository
    # Start from this file and go up to find it
    current_file = Path(__file__).resolve()
    # Go up from src/genbio/leaderboard/main.py to project root
    project_root = current_file.parent
    #datasets_dir = project_root / "datasets"
    return project_root.resolve()


def _load_dataset_module(dataset_name, module_name):
    """Dynamically load a module from the datasets directory."""
    # datasets_dir = _get_datasets_dir()
    dataset_module_name = dataset_name.replace('/', '.').replace('-', '_')
    package_name = f"genbio.datasets.{dataset_module_name}"

    # Build the path to the module file
    with resources.path(package_name, f"{module_name}.py") as module_path:
        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")

        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{module_name}",
            module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
