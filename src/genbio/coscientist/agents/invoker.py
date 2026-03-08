"""Agent invocation: detects, launches, and monitors AI coding agents."""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from genbio.coscientist.adapters.base import DatasetAdapter
from genbio.coscientist.agents.workspace import WorkspaceScaffolder
from genbio.coscientist.config import DataProfile, RunConfig
from genbio.coscientist import console as ui


def detect_agents() -> list[str]:
    """Detect which AI coding agents are installed."""
    agents = []
    if shutil.which("claude"):
        agents.append("claude")
    if shutil.which("codex"):
        agents.append("codex")
    return agents


def select_agent(preference: str = "auto") -> str | None:
    """Select the best available agent."""
    available = detect_agents()
    if not available:
        return None
    if preference != "auto" and preference in available:
        return preference
    if "claude" in available:
        return "claude"
    return available[0]


class AgentInvoker:
    """Orchestrates the agent workflow: scaffold → invoke → validate."""

    def __init__(
        self,
        config: RunConfig,
        adapter: DatasetAdapter,
        profile: DataProfile,
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.profile = profile

    def run(
        self,
        train_data: Any,
        test_data: Any,
        task: Any,
        baseline_score: float | None = None,
    ) -> dict | None:
        """Run the full agent workflow. Returns dict with predictions or None on failure."""
        agent = select_agent(self.config.agent)
        if agent is None:
            ui.error("No AI coding agent found. Install Claude Code (`claude`) or Codex (`codex`).")
            return None

        ui.info(f"Using agent: [bold]{agent}[/bold]")

        scaffolder = WorkspaceScaffolder(self.config, self.adapter, self.profile)
        ws = scaffolder.scaffold(train_data, test_data, baseline_score)
        ui.success(f"Agent workspace: {ws}")

        success = self._invoke_agent(agent, ws)
        if not success:
            ui.error("Agent invocation failed.")
            return None

        return self._validate_results(ws, test_data)

    def _invoke_agent(self, agent: str, workspace: Path) -> bool:
        """Invoke the agent CLI."""
        instructions = (workspace / "INSTRUCTIONS.md").read_text()

        if agent == "claude":
            cmd = [
                "claude", "-p", instructions,
                "--allowedTools", "Read,Edit,Write,Bash",
                "--output-format", "text",
            ]
        elif agent == "codex":
            cmd = [
                "codex", "--full-auto",
                "-C", str(workspace),
                instructions,
            ]
        else:
            ui.error(f"Unknown agent: {agent}")
            return False

        ui.info(f"Invoking {agent}...")
        log_path = workspace / "agent_log.txt"

        try:
            timeout = self._get_timeout()
            with open(log_path, "w") as log_file:
                result = subprocess.run(
                    cmd,
                    cwd=str(workspace),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    text=True,
                )
            if result.returncode == 0:
                ui.success("Agent finished successfully.")
                return True
            else:
                ui.warning(f"Agent exited with code {result.returncode}. Check {log_path}")
                return True  # still try to validate
        except subprocess.TimeoutExpired:
            ui.warning(f"Agent timed out after {timeout}s. Checking partial results.")
            return True
        except FileNotFoundError:
            ui.error(f"Agent binary '{agent}' not found in PATH.")
            return False

    def _validate_results(self, workspace: Path, test_data: Any) -> dict | None:
        """Validate and collect agent outputs."""
        output_dir = workspace / "output"
        predictions_path = output_dir / "predictions.pkl"

        if not predictions_path.exists():
            ui.warning("Agent did not produce predictions.pkl")
            return None

        try:
            with open(predictions_path, "rb") as f:
                raw_preds = pickle.load(f)

            preds = self.adapter.format_predictions(test_data, raw_preds)
            ui.success("Agent predictions validated successfully.")
            return {"predictions": preds, "workspace": workspace}
        except Exception as e:
            ui.error(f"Failed to validate agent predictions: {e}")
            return None

    def _get_timeout(self) -> int:
        from genbio.coscientist.config import BudgetTier
        return {
            BudgetTier.FAST: 300,
            BudgetTier.STANDARD: 600,
            BudgetTier.THOROUGH: 1200,
        }.get(self.config.budget, 600)
