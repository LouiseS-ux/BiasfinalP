"""
Shared pytest fixtures. Loads mock JSON from tests/fixtures/ matching
real data contracts — so tests run without API calls or real pipeline outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def probe_bank() -> list[dict]:
    """Load mock probe_bank.json fixture."""
    with open(FIXTURES_DIR / "probe_bank.json") as f:
        return json.load(f)


@pytest.fixture()
def completions() -> list[dict]:
    """Load mock completions.json fixture."""
    with open(FIXTURES_DIR / "completions.json") as f:
        return json.load(f)


@pytest.fixture()
def predictions() -> list[dict]:
    """Load mock predictions.json fixture."""
    with open(FIXTURES_DIR / "predictions.json") as f:
        return json.load(f)
