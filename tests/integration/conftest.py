"""Pytest configuration for integration tests."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the integration marker to avoid pytest unknown-marker warnings."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring live API access"
        " (deselect with -m 'not integration')",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip integration-marked tests unless -m integration is explicitly passed."""
    marker_expr: str = getattr(config.option, "markexpr", "") or ""
    if "integration" in marker_expr:
        return
    skip = pytest.mark.skip(reason="integration test — run with: pytest -m integration")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)
