"""LLM advisor: interprets data, suggests approaches, polishes reports."""

from __future__ import annotations

import json
from typing import Any

from genbio.coscientist.config import DataProfile
from genbio.coscientist.llm.client import LLMClient, get_llm_client


class Advisor:
    """Language-based advisor backed by an LLM, with template fallbacks."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or get_llm_client()

    @property
    def has_llm(self) -> bool:
        return self.client is not None

    def interpret_profile(self, profile: DataProfile) -> str:
        prompt = (
            f"Summarize this ML dataset for a bioinformatics researcher in 3-4 sentences:\n"
            f"{json.dumps(profile.to_dict(), indent=2)}"
        )
        if self.client:
            return self.client.chat(
                system="You are a bioinformatics ML advisor. Be concise and precise.",
                user=prompt,
            )
        return self._template_interpret_profile(profile)

    def suggest_approach(self, profile: DataProfile, pipeline_names: list[str]) -> str:
        prompt = (
            f"Given this dataset profile, recommend which ML approaches to try and why:\n"
            f"Profile: {json.dumps(profile.to_dict(), indent=2)}\n"
            f"Available pipelines: {pipeline_names}\n"
            f"Give a concise recommendation (3-5 sentences)."
        )
        if self.client:
            return self.client.chat(
                system="You are a bioinformatics ML advisor. Be concise and actionable.",
                user=prompt,
            )
        return self._template_suggest_approach(profile, pipeline_names)

    def interpret_results(self, profile: DataProfile, results: list[dict]) -> str:
        prompt = (
            f"Interpret these ML experiment results for a bioinformatics researcher:\n"
            f"Dataset: {profile.data_type.value}, {profile.task_type.value}\n"
            f"Results: {json.dumps(results, indent=2)}\n"
            f"Provide a concise interpretation (3-5 sentences)."
        )
        if self.client:
            return self.client.chat(
                system="You are a bioinformatics ML advisor. Focus on practical insights.",
                user=prompt,
            )
        return "Results have been recorded. See the metrics table above for comparison."

    def polish_report(self, report_md: str) -> str:
        if not self.client:
            return report_md
        prompt = (
            f"Polish the following ML experiment report. Improve clarity and scientific rigor "
            f"without changing the data or conclusions. Keep the markdown format.\n\n{report_md}"
        )
        return self.client.chat(
            system="You are a scientific writing editor for ML papers. Maintain accuracy.",
            user=prompt,
            max_tokens=4000,
        )

    def chat(self, message: str, context: dict) -> str:
        if self.client:
            system = (
                "You are a bioinformatics ML co-scientist. Help the user with experiment design, "
                "model selection, and interpretation. Be concise.\n"
                f"Context: {json.dumps(context, indent=2, default=str)}"
            )
            return self.client.chat(system=system, user=message)
        return "LLM not available. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY for chat functionality."

    def _template_interpret_profile(self, profile: DataProfile) -> str:
        p = profile
        if p.data_type.value == "dataframe_sequence":
            return (
                f"This is a sequence-based {p.task_type.value} task with {p.train_samples} training "
                f"and {p.test_samples} test samples. Sequences are short ({p.extra.get('seq_len_mean', '?'):.0f}bp avg). "
                f"K-mer featurization with linear models is a strong baseline for this data size."
            )
        elif p.data_type.value == "anndata":
            return (
                f"This is a single-cell expression classification task with {p.train_samples} cells "
                f"and {p.n_features} genes across {p.n_classes} classes. "
                f"PCA + logistic regression with balanced class weights is a robust approach."
            )
        return f"Tabular {p.task_type.value} task with {p.train_samples} samples and {p.n_features} features."

    def _template_suggest_approach(self, profile: DataProfile, pipeline_names: list[str]) -> str:
        return (
            f"Recommended pipelines for this dataset: {', '.join(pipeline_names[:3])}. "
            f"Start with the default configurations, then tune hyperparameters with Optuna."
        )
