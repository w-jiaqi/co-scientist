"""Result validation for agent outputs."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from genbio.coscientist.adapters.base import DatasetAdapter
from genbio.coscientist import console as ui


def validate_agent_output(
    workspace: Path,
    adapter: DatasetAdapter,
    test_data: Any,
    task: Any,
) -> dict | None:
    """Validate agent output and return evaluation metrics if valid."""
    output_dir = workspace / "output"

    predictions_path = output_dir / "predictions.pkl"
    if not predictions_path.exists():
        ui.warning("No predictions.pkl found in agent output.")
        return None

    try:
        with open(predictions_path, "rb") as f:
            raw_preds = pickle.load(f)

        preds = adapter.format_predictions(test_data, raw_preds)
        metrics = task.evaluate(preds, test_data)
        return {"predictions": preds, "metrics": metrics}
    except Exception as e:
        ui.error(f"Validation failed: {e}")
        return None
