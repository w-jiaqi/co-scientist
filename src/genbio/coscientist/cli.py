"""Typer CLI entry point for co-scientist."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from genbio.coscientist.config import BudgetTier, RunConfig
from genbio.coscientist import console as ui

app = typer.Typer(
    name="cosci",
    help="GenBio Co-Scientist: Language-based ML co-scientist CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.command()
def run(
    dataset: Annotated[str, typer.Argument(help="Dataset name (e.g. RNA/translation-efficiency-muscle)")],
    fold: str = typer.Option("0", "--fold", "-f", help="Fold identifier"),
    mode: str = typer.Option("auto", "--mode", "-m", help="Operating mode: interactive or auto"),
    strategy: str = typer.Option("builtin", "--strategy", "-s", help="Strategy: builtin, agentic, or hybrid"),
    budget: str = typer.Option("standard", "--budget", "-b", help="Compute budget: fast, standard, or thorough"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    device: str = typer.Option("auto", "--device", help="Device: cpu, cuda, or auto"),
    output_dir: Path = typer.Option(Path("runs"), "--output-dir", "-o", help="Output directory for runs"),
    user: str = typer.Option("cosci", "--user", "-u", help="Leaderboard username"),
    submit: bool = typer.Option(False, "--submit", help="Submit predictions to leaderboard"),
    agent: str = typer.Option("auto", "--agent", help="Agent to use for agentic/hybrid: claude, codex, or auto"),
) -> None:
    """Run the co-scientist on a dataset: analyze, train, evaluate, and export."""
    config = RunConfig(
        dataset=dataset,
        fold=fold,
        user=user,
        mode=mode,
        strategy=strategy,
        budget=BudgetTier(budget),
        agent=agent,
        seed=seed,
        device=device,
        output_dir=output_dir,
        submit=submit,
    )

    ui.header("GenBio Co-Scientist", f"Dataset: {dataset}  |  Fold: {fold}  |  Strategy: {strategy}")

    from genbio.coscientist.orchestrator import Orchestrator

    orchestrator = Orchestrator(config)
    orchestrator.run()


@app.command()
def list_datasets() -> None:
    """List all available datasets."""
    ui.header("Available Datasets")

    from genbio.coscientist.analyzer import discover_datasets

    datasets = discover_datasets()
    if not datasets:
        ui.warning("No datasets found.")
        return

    for ds in datasets:
        ui.info(ds)


@app.command()
def describe(
    dataset: Annotated[str, typer.Argument(help="Dataset name")],
    fold: str = typer.Option("0", "--fold", "-f", help="Fold identifier"),
) -> None:
    """Describe a dataset with rich formatting."""
    from genbio.leaderboard import BenchmarkTask

    ui.header("Dataset Description", dataset)
    task = BenchmarkTask(name=dataset, fold=fold, user="cosci")
    task.describe()


@app.command()
def predict(
    run_dir: Annotated[Path, typer.Option("--run-dir", help="Path to a completed run directory")],
    input_path: Annotated[Path, typer.Option("--input", "-i", help="Input data file (csv or h5ad)")],
    output_path: Annotated[Path, typer.Option("--output", "-o", help="Output predictions file")] = Path("predictions.csv"),
) -> None:
    """Run inference using a trained model from a previous run."""
    ui.header("Predict", str(run_dir))

    from genbio.coscientist.exporter import run_inference

    run_inference(run_dir, input_path, output_path)
    ui.success(f"Predictions saved to {output_path}")


@app.command()
def report(
    run_dir: Annotated[Path, typer.Option("--run-dir", help="Path to a completed run directory")],
) -> None:
    """Regenerate the report from saved artifacts."""
    ui.header("Report", str(run_dir))

    from genbio.coscientist.reporter import ReportBuilder

    builder = ReportBuilder(run_dir)
    path = builder.generate()
    ui.success(f"Report generated at {path}")


if __name__ == "__main__":
    app()
