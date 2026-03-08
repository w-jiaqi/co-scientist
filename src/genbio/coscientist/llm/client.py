"""Unified LLM client with graceful degradation."""

from __future__ import annotations

import os
from typing import Any


def get_llm_client() -> "LLMClient | None":
    """Return an LLM client if an API key is available, else None."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if anthropic_key:
        try:
            import anthropic
            return AnthropicClient(anthropic_key)
        except ImportError:
            pass

    if openai_key:
        try:
            import openai
            return OpenAIClient(openai_key)
        except ImportError:
            pass

    return None


class LLMClient:
    """Abstract LLM client."""

    def chat(self, system: str, user: str, max_tokens: int = 2000) -> str:
        raise NotImplementedError


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str) -> None:
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

    def chat(self, system: str, user: str, max_tokens: int = 2000) -> str:
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str) -> None:
        import openai
        self.client = openai.OpenAI(api_key=api_key)

    def chat(self, system: str, user: str, max_tokens: int = 2000) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content
