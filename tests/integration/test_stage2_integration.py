"""Integration tests for Stage 2 — one live API call per provider."""

from __future__ import annotations

from typing import Any

import pytest

from src.adapters.anthropic_adapter import AnthropicAdapter
from src.adapters.openai_adapter import OpenAIAdapter
from src.llm_client import CompletionRecord

_PROMPT = "Respond with exactly one word: hello."


@pytest.mark.integration
class TestOpenAILive:
    """Live integration test for the OpenAI adapter."""

    def test_live_call_returns_valid_completion_record(self) -> None:
        """Call OpenAI and assert the response is a valid CompletionRecord."""
        config: dict[str, Any] = {
            "api": {"models": {"openai": "gpt-4o"}, "max_retries": 1}
        }
        adapter = OpenAIAdapter(config)
        record = adapter.complete(prompt=_PROMPT, probe_id="integ_001", gender="male")
        assert isinstance(record, CompletionRecord)
        assert record.probe_id == "integ_001"
        assert record.model == "gpt-4o"
        assert record.completion
        assert record.tokens_used > 0


@pytest.mark.integration
class TestAnthropicLive:
    """Live integration test for the Anthropic adapter."""

    def test_live_call_returns_valid_completion_record(self) -> None:
        """Call Anthropic and assert the response is a valid CompletionRecord."""
        config: dict[str, Any] = {
            "api": {"models": {"anthropic": "claude-opus-4-5"}, "max_retries": 1}
        }
        adapter = AnthropicAdapter(config)
        record = adapter.complete(prompt=_PROMPT, probe_id="integ_001", gender="female")
        assert isinstance(record, CompletionRecord)
        assert record.probe_id == "integ_001"
        assert record.model == "claude-opus-4-5"
        assert record.completion
        assert record.tokens_used > 0
