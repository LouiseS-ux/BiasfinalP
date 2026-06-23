"""Shared LLM client interface, exception type, and CompletionRecord schema."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMError(Exception):
    """Error raised by adapters when any provider API call fails."""


class CompletionRecord(BaseModel):
    """Pydantic model matching the completions.json schema."""

    probe_id: str
    gender: str
    model: str
    prompt: str
    completion: str
    timestamp: str
    tokens_used: int


class LLMClient(ABC):
    """Abstract base class that all provider adapters must implement."""

    @abstractmethod
    def complete(self, prompt: str, probe_id: str, gender: str) -> CompletionRecord:
        """Call the provider API and return a validated CompletionRecord."""
