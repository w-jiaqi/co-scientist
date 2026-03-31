"""Agent invocation: detects, launches, and monitors AI coding agents."""

from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
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
    """Orchestrates the agent workflow: scaffold -> invoke -> validate."""

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
        """Invoke the agent CLI.

        Uses os.system() to run through the user's login shell, which ensures
        the correct PATH (NVM, etc.), full network access, and proper TTY
        handling that claude/codex require.

        In interactive mode: the agent prompts the user for command approvals,
        so you can review/reject/edit each step. In auto mode: all commands
        are auto-approved for fully hands-off operation.
        """
        interactive = self.config.mode == "interactive"
        timeout = self._get_timeout()

        if agent == "claude":
            if interactive:
                shell_cmd = (
                    f'cd "{workspace}" && '
                    f'claude "$(cat INSTRUCTIONS.md)" '
                )
            else:
                shell_cmd = (
                    f'cd "{workspace}" && '
                    f'claude -p "$(cat INSTRUCTIONS.md)" '
                    f'--allowedTools "Bash,Read,Edit,Write" '
                    f'2>&1 | tee agent_log.txt'
                )
        elif agent == "codex":
            if interactive:
                shell_cmd = (
                    f'cd "{workspace}" && '
                    f'codex '
                    f'"$(cat INSTRUCTIONS.md)" '
                )
            else:
                shell_cmd = (
                    f'cd "{workspace}" && '
                    f'codex --ask-for-approval never '
                    f'--no-alt-screen '
                    f'"$(cat INSTRUCTIONS.md)" '
                    f'2>&1 | tee agent_log.txt'
                )
        else:
            ui.error(f"Unknown agent: {agent}")
            return False

        mode_label = "interactive (you control approvals)" if interactive else "autonomous"
        ui.info(f"Invoking {agent} in {mode_label} mode...")
        ui.console.print("[dim]" + "─" * 60 + "[/dim]")

        rc = os.system(shell_cmd)

        ui.console.print("[dim]" + "─" * 60 + "[/dim]")

        if rc == 0:
            ui.success("Agent finished successfully.")
        else:
            ui.warning(f"Agent exited with code {rc}.")

        return True

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


def test_agent_connection(agent: str = "auto") -> bool:
    """Quick test: can we invoke the agent and get a response?
    
    Run from your terminal: python -c "from genbio.coscientist.agents.invoker import test_agent_connection; test_agent_connection()"
    """
    selected = select_agent(agent)
    if selected is None:
        print("No agent found. Install claude or codex CLI.")
        return False

    print(f"Testing {selected}...")

    if selected == "claude":
        rc = os.system('claude -p "Reply with OK"')
    elif selected == "codex":
        rc = os.system('codex --ask-for-approval never --no-alt-screen "Reply with OK"')
    else:
        print(f"Unknown agent: {selected}")
        return False

    ok = rc == 0
    print(f"\nResult: {'PASS' if ok else 'FAIL'} (exit code {rc})")
    return ok
