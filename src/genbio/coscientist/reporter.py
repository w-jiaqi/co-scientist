"""Report generation: Markdown reports with optional figures."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from genbio.coscientist.config import DataProfile, ExperimentResult, TaskType


class ReportBuilder:
    """Generates a Markdown report from run artifacts."""

    def __init__(
        self,
        run_dir: Path,
        profile: DataProfile | None = None,
        results: list[ExperimentResult] | None = None,
        test_metrics: dict | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.profile = profile or self._load_profile()
        self.results = results or []
        self.test_metrics = test_metrics or self._load_test_metrics()

    def _load_profile(self) -> DataProfile:
        path = self.run_dir / "data_profile.json"
        if path.exists():
            d = json.loads(path.read_text())
            from genbio.coscientist.config import DataType
            d["data_type"] = DataType(d["data_type"])
            d["task_type"] = TaskType(d["task_type"])
            return DataProfile(**d)
        raise FileNotFoundError(f"No data_profile.json in {self.run_dir}")

    def _load_test_metrics(self) -> dict:
        path = self.run_dir / "test_metrics.json"
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def generate(self) -> Path:
        """Generate the report and write to run_dir/report.md."""
        sections = [
            self._executive_summary(),
            self._dataset_profile(),
            self._methodology(),
            self._results_section(),
            self._analysis(),
            self._reproducibility(),
            self._recommendations(),
        ]

        report = "\n\n---\n\n".join(sections)
        path = self.run_dir / "report.md"
        path.write_text(report)

        self._generate_figures()

        return path

    def _executive_summary(self) -> str:
        p = self.profile
        best = max(self.results, key=lambda r: r.primary_metric_value) if self.results else None
        primary = self.test_metrics.get("primary_metric", "")
        primary_val = self.test_metrics.get(primary, 0.0) if primary else 0.0

        lines = [
            "# Co-Scientist Report",
            "",
            "## Executive Summary",
            "",
            f"- **Dataset**: {p.data_type.value}",
            f"- **Task**: {p.task_type.value}",
            f"- **Train samples**: {p.train_samples}",
            f"- **Test samples**: {p.test_samples}",
        ]
        if best:
            lines.append(f"- **Best model**: {best.name} ({best.pipeline_name})")
            lines.append(f"- **Validation {best.primary_metric_name}**: {best.primary_metric_value:.4f}")
        if primary:
            lines.append(f"- **Test {primary}**: {primary_val:.4f}")

        lines.append(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        return "\n".join(lines)

    def _dataset_profile(self) -> str:
        p = self.profile
        lines = [
            "## Dataset Profile",
            "",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| Data type | {p.data_type.value} |",
            f"| Task type | {p.task_type.value} |",
            f"| Train samples | {p.train_samples} |",
            f"| Test samples | {p.test_samples} |",
        ]
        if p.n_features is not None:
            lines.append(f"| Features | {p.n_features} |")
        if p.n_classes is not None:
            lines.append(f"| Classes | {p.n_classes} |")
        if p.sequence_column:
            lines.append(f"| Sequence column | {p.sequence_column} |")
            for k, v in p.extra.items():
                if k.startswith("seq_"):
                    lines.append(f"| {k} | {v} |")

        if p.class_distribution:
            lines.extend(["", "### Class Distribution", ""])
            lines.append("| Class | Count |")
            lines.append("|-------|-------|")
            for cls, cnt in sorted(p.class_distribution.items(), key=lambda x: -x[1]):
                lines.append(f"| {cls} | {cnt} |")

        if p.label_stats:
            lines.extend(["", "### Label Statistics", ""])
            lines.append("| Stat | Value |")
            lines.append("|------|-------|")
            for k, v in p.label_stats.items():
                lines.append(f"| {k} | {v:.4f} |")

        return "\n".join(lines)

    def _methodology(self) -> str:
        lines = [
            "## Methodology",
            "",
            "### Approach",
            "",
        ]

        if self.profile.data_type.value == "dataframe_sequence":
            lines.append("DNA/RNA sequences were featurized using character-level k-mer counting (CountVectorizer). "
                          "Additional engineered features (GC content, sequence length) were included. "
                          "Models were trained with cross-validation using the existing fold structure.")
        elif self.profile.data_type.value == "anndata":
            lines.append("Single-cell expression data was preprocessed with normalization (total count 10k), "
                          "log1p transformation, and PCA dimensionality reduction. "
                          "Classifiers were configured with balanced class weights to optimize macro F1.")
        else:
            lines.append("Tabular features were standardized and fed into various model families.")

        if self.results:
            lines.extend(["", "### Models Attempted", ""])
            lines.append("| # | Model | Pipeline | CV Score |")
            lines.append("|---|-------|----------|----------|")
            for i, r in enumerate(self.results):
                lines.append(f"| {i+1} | {r.name} | {r.pipeline_name} | {r.primary_metric_value:.4f} |")

        return "\n".join(lines)

    def _results_section(self) -> str:
        lines = ["## Results", ""]

        if self.test_metrics:
            primary = self.test_metrics.get("primary_metric", "")
            lines.append("### Test Set Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in self.test_metrics.items():
                if k == "primary_metric":
                    continue
                marker = " **" if k == primary else ""
                lines.append(f"| {k}{marker} | {v:.6f} |")

        if (self.run_dir / "figures").exists():
            lines.extend(["", "### Figures", ""])
            for fig in sorted((self.run_dir / "figures").glob("*.png")):
                lines.append(f"![{fig.stem}](figures/{fig.name})")

        return "\n".join(lines)

    def _analysis(self) -> str:
        lines = ["## Analysis", ""]

        best = max(self.results, key=lambda r: r.primary_metric_value) if self.results else None
        if best:
            lines.append(f"The best performing model was **{best.name}** using the {best.pipeline_name} pipeline, "
                          f"achieving a cross-validation {best.primary_metric_name} of {best.primary_metric_value:.4f}.")

        if self.profile.task_type == TaskType.REGRESSION:
            lines.append("")
            lines.append("For regression tasks, Spearman correlation captures rank-order accuracy, "
                          "which is robust to monotonic transformations of predictions.")
        else:
            lines.append("")
            lines.append("Macro F1 was used as the primary metric, giving equal weight to all classes "
                          "regardless of their frequency in the dataset.")

        return "\n".join(lines)

    def _reproducibility(self) -> str:
        lines = [
            "## Reproducibility",
            "",
            "### Configuration",
            "",
            "The full configuration is saved in `config.yaml` in the run directory.",
            "",
            "### How to Reproduce",
            "",
            "```bash",
            "# Re-run the exact same experiment",
            f"cosci run --dataset <dataset> --fold <fold> --budget <budget> --seed <seed>",
            "```",
            "",
            "### Exported Project",
            "",
            "A standalone project is available in `exported_project/` with:",
            "- `train.py`: re-train the model from scratch",
            "- `predict.py`: run inference on new data",
            "- `requirements.txt`: pinned dependencies",
        ]
        return "\n".join(lines)

    def _recommendations(self) -> str:
        lines = [
            "## Recommendations",
            "",
            "With additional compute and time, consider:",
            "",
        ]

        if self.profile.data_type.value == "dataframe_sequence":
            lines.extend([
                "- Trying longer k-mer ranges (up to 8-mers)",
                "- Adding position-specific k-mer features",
                "- Using pretrained DNA language model embeddings (e.g. DNABERT, AIDO.DNA)",
                "- Neural network models (1D CNN on one-hot encoded sequences)",
            ])
        elif self.profile.data_type.value == "anndata":
            lines.extend([
                "- Increasing the number of highly variable genes",
                "- Using scVI or other deep generative models for representation learning",
                "- Trying pretrained cell foundation model embeddings (e.g. AIDO.Cell)",
                "- Feature selection via LASSO or mutual information",
            ])
        else:
            lines.extend([
                "- More extensive hyperparameter search",
                "- Feature engineering based on domain knowledge",
                "- Deep learning approaches (if dataset is large enough)",
            ])

        lines.extend([
            "",
            "For the best results, use `--strategy hybrid` to combine built-in pipelines "
            "with an AI coding agent (Claude Code or Codex).",
        ])

        return "\n".join(lines)

    def _generate_figures(self) -> None:
        """Generate plots and save to run_dir/figures/."""
        fig_dir = self.run_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(style="whitegrid")
        except ImportError:
            return

        # Label distribution
        if self.profile.class_distribution:
            self._plot_class_distribution(fig_dir, plt)
        elif self.profile.label_stats:
            self._plot_label_histogram(fig_dir, plt)

        # Validation metrics comparison
        if len(self.results) >= 2:
            self._plot_metrics_comparison(fig_dir, plt)

        plt.close("all")

    def _plot_class_distribution(self, fig_dir: Path, plt) -> None:
        classes = list(self.profile.class_distribution.keys())
        counts = list(self.profile.class_distribution.values())

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(classes, counts, color="#4C9BE8")
        ax.set_xlabel("Count")
        ax.set_title("Class Distribution (Training Set)")
        fig.tight_layout()
        fig.savefig(fig_dir / "label_distribution.png", dpi=150)

    def _plot_label_histogram(self, fig_dir: Path, plt) -> None:
        stats = self.profile.label_stats
        if not stats:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5,
                f"Mean: {stats['mean']:.4f}\nStd: {stats['std']:.4f}\n"
                f"Min: {stats['min']:.4f}\nMax: {stats['max']:.4f}",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_title("Label Statistics (Training Set)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / "label_distribution.png", dpi=150)

    def _plot_metrics_comparison(self, fig_dir: Path, plt) -> None:
        names = [r.name for r in self.results]
        scores = [r.primary_metric_value for r in self.results]
        metric_name = self.results[0].primary_metric_name

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#4C9BE8" if s < max(scores) else "#2ECC71" for s in scores]
        ax.barh(names, scores, color=colors)
        ax.set_xlabel(metric_name)
        ax.set_title(f"Model Comparison (CV {metric_name})")
        fig.tight_layout()
        fig.savefig(fig_dir / "validation_metrics.png", dpi=150)
