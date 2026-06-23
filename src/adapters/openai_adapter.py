"""OpenAI GPT-4o adapter implementing LLMClient."""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

import openai

from src.llm_client import CompletionRecord, LLMClient, LLMError

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMClient):
    """LLMClient implementation that calls the OpenAI Chat Completions API."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the adapter, loading API key and config values."""
        self._api_key: str = os.environ["OPENAI_API_KEY"]
        self._model: str = config["api"]["models"]["openai"]
        self._max_retries: int = config["api"]["max_retries"]
        self._client = openai.OpenAI(api_key=self._api_key)

    def complete(self, prompt: str, probe_id: str, gender: str) -> CompletionRecord:
        """Send a prompt to OpenAI and return a CompletionRecord, with retry on 429."""
        delay = 1.0
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                )
                tokens = response.usage.total_tokens if response.usage else 0
                logger.info(
                    "OpenAI %s probe=%s gender=%s tokens=%d",
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
                    completion=response.choices[0].message.content or "",
                    timestamp=datetime.now(UTC).isoformat(),
                    tokens_used=tokens,
                )
            except openai.RateLimitError as exc:
                if attempt < self._max_retries:
                    logger.warning("OpenAI rate limit, retrying in %.1fs", delay)
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise LLMError(
                        f"OpenAI rate limit after {self._max_retries} retries"
                    ) from exc
            except openai.OpenAIError as exc:
                raise LLMError(f"OpenAI API error: {exc}") from exc
        raise LLMError("OpenAI retries exhausted")  # pragma: no cover
