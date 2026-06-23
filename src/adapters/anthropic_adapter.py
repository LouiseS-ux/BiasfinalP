"""Anthropic claude-opus-4-5 adapter implementing LLMClient."""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

import anthropic

from src.llm_client import CompletionRecord, LLMClient, LLMError

logger = logging.getLogger(__name__)

_MAX_TOKENS = 1024


class AnthropicAdapter(LLMClient):
    """LLMClient implementation that calls the Anthropic Messages API."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the adapter, loading API key and config values."""
        self._api_key: str = os.environ["ANTHROPIC_API_KEY"]
        self._model: str = config["api"]["models"]["anthropic"]
        self._max_retries: int = config["api"]["max_retries"]
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def complete(self, prompt: str, probe_id: str, gender: str) -> CompletionRecord:
        """Call Anthropic and return a CompletionRecord; retries on 429."""
        delay = 1.0
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                tokens = response.usage.input_tokens + response.usage.output_tokens
                logger.info(
                    "Anthropic %s probe=%s gender=%s tokens=%d",
                    self._model,
                    probe_id,
                    gender,
                    tokens,
                )
                return CompletionRecord(
                    probe_id=probe_id,
                    gender=gender,
                    model=self._model,
                    prompt=prompt,
                    completion=response.content[0].text if response.content else "",
                    timestamp=datetime.now(UTC).isoformat(),
                    tokens_used=tokens,
                )
            except anthropic.RateLimitError as exc:
                if attempt < self._max_retries:
                    logger.warning("Anthropic rate limit, retrying in %.1fs", delay)
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise LLMError(
                        f"Anthropic rate limit after {self._max_retries} retries"
                    ) from exc
            except anthropic.APIError as exc:
                raise LLMError(f"Anthropic API error: {exc}") from exc
        raise LLMError("Anthropic retries exhausted")  # pragma: no cover
