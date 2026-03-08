"""Rich console helpers for co-scientist CLI output."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.markdown import Markdown
from rich import box

console = Console()


def header(title: str, subtitle: str = "") -> None:
    text = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        text += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(text, box=box.DOUBLE, expand=False))


def section(title: str) -> None:
    console.print(f"\n[bold yellow]{title}[/bold yellow]")
    console.print("[dim]" + "─" * 60 + "[/dim]")


def info(msg: str) -> None:
    console.print(f"  [cyan]ℹ[/cyan]  {msg}")


def success(msg: str) -> None:
    console.print(f"  [green]✓[/green]  {msg}")


def warning(msg: str) -> None:
    console.print(f"  [yellow]⚠[/yellow]  {msg}")


def error(msg: str) -> None:
    console.print(f"  [red]✗[/red]  {msg}")


def step(number: int, msg: str) -> None:
    console.print(f"\n  [bold white][{number}][/bold white] {msg}")


def metrics_table(metrics: dict[str, float], primary_metric: str, title: str = "Metrics") -> None:
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="white")
    table.add_column("", style="yellow")

    for name, value in metrics.items():
        if name == "primary_metric":
            continue
        marker = "★" if name == primary_metric else ""
        table.add_row(name, f"{value:.6f}", marker)

    console.print(table)


def experiments_table(results: list, title: str = "Experiment Results") -> None:
    if not results:
        return
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("#", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Pipeline", style="white")
    table.add_column("Primary Metric", justify="right", style="green")
    table.add_column("Time", justify="right", style="dim")

    for i, r in enumerate(results):
        table.add_row(
            str(i),
            r.name,
            r.pipeline_name,
            f"{r.primary_metric_value:.6f}",
            f"{r.duration_seconds:.1f}s",
        )

    console.print(table)


def data_profile_panel(profile) -> None:
    from genbio.coscientist.config import DataProfile

    lines = [
        f"Data type: [cyan]{profile.data_type.value}[/cyan]",
        f"Task type: [cyan]{profile.task_type.value}[/cyan]",
        f"Train samples: [white]{profile.train_samples}[/white]",
        f"Test samples: [white]{profile.test_samples}[/white]",
        f"Label column: [white]{profile.label_column}[/white]",
    ]
    if profile.n_features is not None:
        lines.append(f"Features: [white]{profile.n_features}[/white]")
    if profile.n_classes is not None:
        lines.append(f"Classes: [white]{profile.n_classes}[/white]")
    if profile.sequence_column:
        lines.append(f"Sequence column: [white]{profile.sequence_column}[/white]")

    console.print(Panel("\n".join(lines), title="Dataset Profile", box=box.ROUNDED))


def spinner_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    )
