"""Unit tests for Stage 2 LLM query runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import anthropic
import openai
import pytest

from src.adapters.anthropic_adapter import AnthropicAdapter
from src.adapters.openai_adapter import OpenAIAdapter
from src.llm_client import CompletionRecord, LLMError
from src.stage2_query_llm import run_queries

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

_CONFIG_OPENAI: dict[str, Any] = {
    "api": {"models": {"openai": "gpt-4o"}, "max_retries": 1}
}
_CONFIG_ANTHROPIC: dict[str, Any] = {
    "api": {"models": {"anthropic": "claude-opus-4-5"}, "max_retries": 1}
}


def _make_record(**overrides: str | int) -> CompletionRecord:
    defaults: dict[str, Any] = {
        "probe_id": "wb_001",
        "gender": "male",
        "model": "gpt-4o",
        "prompt": "A man in a meeting.",
        "completion": "He sat at the table.",
        "timestamp": "2024-01-15T10:30:00+00:00",
        "tokens_used": 20,
    }
    defaults.update(overrides)
    return CompletionRecord(**defaults)


def _probe() -> dict[str, str]:
    return {
        "probe_id": "wb_001",
        "male_prompt": "A man in a meeting.",
        "female_prompt": "A woman in a meeting.",
    }


class _FakeAnthropicError(anthropic.APIError):
    """Minimal anthropic.APIError subclass that bypasses the SDK constructor."""

    def __init__(self) -> None:
        Exception.__init__(self, "test error")


class _FakeAnthropicRateLimitError(anthropic.RateLimitError):
    """Minimal RateLimitError subclass for testing retry behaviour."""

    def __init__(self) -> None:
        Exception.__init__(self, "rate limited")


class TestCompletionRecordSchema:
    """Tests that completions_fixture.json entries satisfy the Section 5.2 schema."""

    def test_fixture_entries_are_valid(self) -> None:
        with open(_FIXTURES_DIR / "completions_fixture.json") as f:
            entries: list[dict[str, Any]] = json.load(f)
        required = {
            "probe_id",
            "gender",
            "model",
            "prompt",
            "completion",
            "timestamp",
            "tokens_used",
        }
        for entry in entries:
            assert set(entry.keys()) == required
            record = CompletionRecord(**entry)
            assert isinstance(record.tokens_used, int)

    def test_fixture_has_one_entry_per_model(self) -> None:
        with open(_FIXTURES_DIR / "completions_fixture.json") as f:
            entries: list[dict[str, Any]] = json.load(f)
        models = {e["model"] for e in entries}
        assert "gpt-4o" in models
        assert "claude-opus-4-5" in models


class TestOpenAIAdapter:
    """Tests for OpenAIAdapter using a mocked OpenAI client."""

    def test_complete_returns_correct_completion_record(self) -> None:
        mock_response = MagicMock()
        mock_response.usage.total_tokens = 42
        mock_response.choices[0].message.content = "He sat at the table."

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.OpenAI") as mock_cls,
        ):
            mock_cls.return_value.chat.completions.create.return_value = mock_response
            adapter = OpenAIAdapter(_CONFIG_OPENAI)
            record = adapter.complete("A man in a meeting.", "wb_001", "male")

        assert isinstance(record, CompletionRecord)
        assert record.probe_id == "wb_001"
        assert record.gender == "male"
        assert record.model == "gpt-4o"
        assert record.prompt == "A man in a meeting."
        assert record.completion == "He sat at the table."
        assert record.tokens_used == 42

    def test_raises_llm_error_not_openai_exception(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.OpenAI") as mock_cls,
        ):
            mock_cls.return_value.chat.completions.create.side_effect = (
                openai.OpenAIError("provider failure")
            )
            adapter = OpenAIAdapter(_CONFIG_OPENAI)
            with pytest.raises(LLMError):
                adapter.complete("prompt", "wb_001", "male")

    def test_retries_on_429_then_succeeds(self) -> None:
        mock_response = MagicMock()
        mock_response.usage.total_tokens = 10
        mock_response.choices[0].message.content = "Retry success."
        call_count = 0

        def _side_effect(*args: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise openai.RateLimitError(
                    message="rate limited", response=MagicMock(), body=None
                )
            return mock_response

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch("openai.OpenAI") as mock_cls,
            patch("src.adapters.openai_adapter.time.sleep") as mock_sleep,
        ):
            mock_cls.return_value.chat.completions.create.side_effect = _side_effect
            adapter = OpenAIAdapter(_CONFIG_OPENAI)
            record = adapter.complete("A man in a meeting.", "wb_001", "male")

        assert isinstance(record, CompletionRecord)
        assert mock_cls.return_value.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once()


class TestAnthropicAdapter:
    """Tests for AnthropicAdapter using a mocked Anthropic client."""

    def test_complete_returns_correct_completion_record(self) -> None:
        mock_text = MagicMock()
        mock_text.text = "She led the meeting."
        mock_response = MagicMock()
        mock_response.content = [mock_text]
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 12

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as mock_cls,
        ):
            mock_cls.return_value.messages.create.return_value = mock_response
            adapter = AnthropicAdapter(_CONFIG_ANTHROPIC)
            record = adapter.complete("A woman in a meeting.", "wb_002", "female")

        assert isinstance(record, CompletionRecord)
        assert record.probe_id == "wb_002"
        assert record.gender == "female"
        assert record.model == "claude-opus-4-5"
        assert record.completion == "She led the meeting."
        assert record.tokens_used == 27

    def test_raises_llm_error_not_anthropic_exception(self) -> None:
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as mock_cls,
        ):
            mock_cls.return_value.messages.create.side_effect = _FakeAnthropicError()
            adapter = AnthropicAdapter(_CONFIG_ANTHROPIC)
            with pytest.raises(LLMError):
                adapter.complete("prompt", "wb_002", "female")

    def test_retries_on_429_then_succeeds(self) -> None:
        mock_text = MagicMock()
        mock_text.text = "Retry success."
        mock_response = MagicMock()
        mock_response.content = [mock_text]
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 5
        call_count = 0

        def _side_effect(*args: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeAnthropicRateLimitError()
            return mock_response

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}),
            patch("anthropic.Anthropic") as mock_cls,
            patch("src.adapters.anthropic_adapter.time.sleep") as mock_sleep,
        ):
            mock_cls.return_value.messages.create.side_effect = _side_effect
            adapter = AnthropicAdapter(_CONFIG_ANTHROPIC)
            record = adapter.complete("A woman in a meeting.", "wb_002", "female")

        assert isinstance(record, CompletionRecord)
        assert mock_cls.return_value.messages.create.call_count == 2
        mock_sleep.assert_called_once()


class TestRunQueries:
    """Tests for run_queries: resume logic, write-after-each, and error handling."""

    def test_skips_already_collected_triple(self, tmp_path: Path) -> None:
        existing = [
            _make_record(gender="male").model_dump(),
            _make_record(gender="female").model_dump(),
        ]
        mock_client = MagicMock()
        completions_path = tmp_path / "completions.json"

        _, skipped, _ = run_queries(
            probes=[_probe()],
            clients={"gpt-4o": mock_client},
            completions=list(existing),
            completions_path=completions_path,
            delay=0.0,
        )

        assert skipped == 2
        mock_client.complete.assert_not_called()

    def test_writes_completions_after_each_call(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = _make_record()
        completions_path = tmp_path / "completions.json"

        run_queries(
            probes=[_probe()],
            clients={"gpt-4o": mock_client},
            completions=[],
            completions_path=completions_path,
            delay=0.0,
        )

        assert completions_path.exists()
        with open(completions_path) as f:
            data: list[dict[str, Any]] = json.load(f)
        assert len(data) == 2  # male + female

    def test_llm_error_is_counted_not_raised(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.complete.side_effect = LLMError("API failed")
        completions_path = tmp_path / "completions.json"

        collected, _, failures = run_queries(
            probes=[_probe()],
            clients={"gpt-4o": mock_client},
            completions=[],
            completions_path=completions_path,
            delay=0.0,
        )

        assert collected == 0
        assert failures == 2  # male + female both fail

    def test_written_json_matches_completion_record_schema(
        self, tmp_path: Path
    ) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = _make_record()
        completions_path = tmp_path / "completions.json"

        run_queries(
            probes=[_probe()],
            clients={"gpt-4o": mock_client},
            completions=[],
            completions_path=completions_path,
            delay=0.0,
        )

        with open(completions_path) as f:
            data: list[dict[str, Any]] = json.load(f)
        required = {
            "probe_id",
            "gender",
            "model",
            "prompt",
            "completion",
            "timestamp",
            "tokens_used",
        }
        for entry in data:
            assert set(entry.keys()) == required
            CompletionRecord(**entry)
