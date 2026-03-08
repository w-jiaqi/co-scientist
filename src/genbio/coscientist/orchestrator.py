"""Main orchestration loop: the co-scientist brain."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

from genbio.coscientist.config import RunConfig, ExperimentResult
from genbio.coscientist import console as ui


class Orchestrator:
    """Drives the full co-scientist workflow: load → analyze → plan → run → export → report."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config

    def run(self) -> None:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
        warnings.filterwarnings("ignore", message=".*Liblinear.*")

        config = self.config
        config.run_dir.mkdir(parents=True, exist_ok=True)
        config.save()

        # 1. Load task
        ui.step(1, "Loading dataset...")
        from genbio.leaderboard import BenchmarkTask

        task = BenchmarkTask(name=config.dataset, fold=config.fold, user=config.user)
        train_data, test_data = task.setup()
        ui.success(f"Loaded {config.dataset} fold {config.fold}")

        # 2. Analyze dataset
        ui.step(2, "Analyzing dataset...")
        from genbio.coscientist.analyzer import analyze_dataset

        adapter, profile = analyze_dataset(train_data, test_data)
        profile.save(config.run_dir / "data_profile.json")
        ui.data_profile_panel(profile)

        # 3. Strategy dispatch
        if config.strategy == "agentic":
            self._run_agentic(config, task, adapter, profile, train_data, test_data)
            return

        # Built-in / hybrid path
        best_result, plan, all_results = self._run_builtin(config, adapter, profile, train_data, test_data, task)

        if config.strategy == "hybrid":
            self._run_hybrid_agent(config, task, adapter, profile, train_data, test_data, best_result)

        # Generate test predictions and evaluate
        ui.step(7, "Generating test predictions...")
        from genbio.coscientist.runner import ExperimentRunner

        runner = ExperimentRunner(config, adapter, profile)
        runner.results = all_results
        preds = runner.generate_test_predictions(best_result, plan, train_data, test_data)

        ui.step(8, "Evaluating on test set...")
        test_metrics = task.evaluate(preds, test_data)
        primary_metric = test_metrics.get("primary_metric", "unknown")
        ui.metrics_table(test_metrics, primary_metric, title="Test Set Results")

        # Save test metrics
        test_metrics_path = config.run_dir / "test_metrics.json"
        test_metrics_path.write_text(json.dumps(test_metrics, indent=2))

        # Submit if requested
        if config.submit:
            ui.step(9, "Submitting to leaderboard...")
            task.submit(preds, name=f"cosci_{best_result.name}", description=f"Co-Scientist {config.strategy} run")

        # Export and report
        self._export_and_report(config, profile, best_result, test_metrics, plan, adapter, train_data, test_data, all_results)

        ui.console.print()
        ui.success(f"Run complete! Output: [bold]{config.run_dir}[/bold]")

    def _run_builtin(self, config, adapter, profile, train_data, test_data, task):
        """Run built-in pipeline experiments."""
        from genbio.coscientist.planner import create_plan
        from genbio.coscientist.runner import ExperimentRunner

        # Plan
        ui.step(3, "Planning experiments...")
        plan = create_plan(profile, config.budget)

        if config.mode == "interactive":
            self._interactive_plan_review(plan)

        ui.info(plan.reasoning)
        for i, exp in enumerate(plan.experiments):
            tune_str = f"(tune {exp.optuna_trials} trials)" if exp.tune else "(default config)"
            ui.info(f"  {i+1}. {exp.name} {tune_str}")
        if plan.do_ensemble:
            ui.info("  + Ensemble of top models")

        # Execute
        ui.section("Execution")
        runner = ExperimentRunner(config, adapter, profile)
        results = runner.execute(plan, train_data, test_data)

        # Select best
        ui.section("Results")
        ui.experiments_table(results)
        best = runner.select_best()
        ui.success(f"Best model: [bold]{best.name}[/bold] ({best.primary_metric_name} = {best.primary_metric_value:.4f})")

        if config.mode == "interactive":
            self._interactive_results_review(results, best)

        return best, plan, results

    def _run_agentic(self, config, task, adapter, profile, train_data, test_data):
        """Full agentic delegation."""
        ui.step(3, "Delegating to AI coding agent...")
        try:
            from genbio.coscientist.agents.invoker import AgentInvoker

            invoker = AgentInvoker(config, adapter, profile)
            agent_result = invoker.run(train_data, test_data, task)

            if agent_result is not None:
                ui.success("Agent completed successfully")
                preds = agent_result["predictions"]
                test_metrics = task.evaluate(preds, test_data)
                primary_metric = test_metrics.get("primary_metric", "unknown")
                ui.metrics_table(test_metrics, primary_metric, title="Agent Results")

                test_metrics_path = config.run_dir / "test_metrics.json"
                test_metrics_path.write_text(json.dumps(test_metrics, indent=2))

                if config.submit:
                    task.submit(preds, name="cosci_agent", description="Co-Scientist agentic run")
            else:
                ui.warning("Agent did not produce valid results. No fallback in agentic-only mode.")
        except Exception as e:
            ui.error(f"Agent failed: {e}")
            ui.warning("No fallback available in agentic-only mode.")

    def _run_hybrid_agent(self, config, task, adapter, profile, train_data, test_data, baseline_result):
        """Hybrid mode: try agent after establishing a baseline."""
        ui.section("Hybrid: Agentic Phase")
        ui.step(6, f"Invoking agent to beat baseline ({baseline_result.primary_metric_value:.4f})...")

        try:
            from genbio.coscientist.agents.invoker import AgentInvoker

            invoker = AgentInvoker(config, adapter, profile)
            agent_result = invoker.run(train_data, test_data, task, baseline_score=baseline_result.primary_metric_value)

            if agent_result is not None:
                preds = agent_result["predictions"]
                test_metrics = task.evaluate(preds, test_data)
                primary_metric = test_metrics.get("primary_metric", "unknown")
                agent_score = test_metrics.get(primary_metric, 0.0)

                if agent_score > baseline_result.primary_metric_value:
                    ui.success(f"Agent beat baseline: {agent_score:.4f} > {baseline_result.primary_metric_value:.4f}")
                else:
                    ui.info(f"Agent score ({agent_score:.4f}) did not beat baseline ({baseline_result.primary_metric_value:.4f}). Using built-in model.")
            else:
                ui.info("Agent did not produce valid results. Using built-in model.")
        except Exception as e:
            ui.warning(f"Agent phase failed: {e}. Using built-in model.")

    def _export_and_report(self, config, profile, best_result, test_metrics, plan, adapter, train_data, test_data, all_results=None):
        """Generate report and export portable project."""
        from genbio.coscientist.reporter import ReportBuilder
        from genbio.coscientist.exporter import ProjectExporter

        ui.section("Output")

        # Report
        ui.step(10, "Generating report...")
        results_list = all_results if all_results else [best_result]
        reporter = ReportBuilder(config.run_dir, profile, results_list, test_metrics)
        report_path = reporter.generate()
        ui.success(f"Report: {report_path}")

        # Export
        ui.step(11, "Exporting portable project...")
        exporter = ProjectExporter(config, profile, best_result, plan)
        export_path = exporter.export()
        ui.success(f"Exported project: {export_path}")

    def _interactive_plan_review(self, plan) -> None:
        """Let the user review and adjust the plan in interactive mode."""
        from rich.prompt import Confirm, Prompt

        ui.console.print()
        ui.info(plan.summary())
        ui.console.print()

        while True:
            action = Prompt.ask(
                "What would you like to do?",
                choices=["proceed", "ask", "quit"],
                default="proceed",
            )
            if action == "proceed":
                break
            elif action == "ask":
                question = Prompt.ask("Your question")
                from genbio.coscientist.llm.advisor import Advisor
                advisor = Advisor()
                answer = advisor.chat(question, {"plan": plan.summary()})
                ui.console.print(f"\n[dim]{answer}[/dim]\n")
            else:
                ui.info("Run cancelled by user.")
                raise SystemExit(0)

    def _interactive_results_review(self, results, best) -> None:
        """Let the user review results in interactive mode."""
        from rich.prompt import Confirm, Prompt

        ui.console.print()
        while True:
            action = Prompt.ask(
                "What would you like to do?",
                choices=["accept", "ask", "quit"],
                default="accept",
            )
            if action == "accept":
                break
            elif action == "ask":
                question = Prompt.ask("Your question")
                from genbio.coscientist.llm.advisor import Advisor
                advisor = Advisor()
                result_data = [{"name": r.name, "score": r.primary_metric_value} for r in results]
                answer = advisor.chat(question, {"results": result_data, "best": best.name})
                ui.console.print(f"\n[dim]{answer}[/dim]\n")
            else:
                ui.info("Run ended by user. Artifacts saved in the run directory.")
                raise SystemExit(0)
