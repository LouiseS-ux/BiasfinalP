"""
Stage 1 — Probe Bank Validator.

Responsibility: Loads the curated probe bank from data/probe_bank.json,
validates every entry against the ProbeEntry schema, checks structural
invariants (count, uniqueness, field presence, allowed values), and logs
a summary.


"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path("config.yaml")

_EXPECTED_TOTAL = 150
_VALID_SOURCES: frozenset[str] = frozenset({"winobias", "original"})
_VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "professional_role",
        "personality_trait",
        "ambiguous_scenario",
        "coreference_ambiguity",
    }
)


class ProbeEntry(BaseModel):
    probe_id: str
    category: str
    role: str
    source: str
    male_prompt: str
    female_prompt: str


def _load_config() -> dict[str, Any]:
    """Load config.yaml and return it as a dictionary."""
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_probe_bank(path: Path) -> list[ProbeEntry]:
    """Load probe_bank.json and validate each entry against the ProbeEntry schema."""
    with open(path) as f:
        raw: list[dict[str, Any]] = json.load(f)
    probes: list[ProbeEntry] = []
    for i, record in enumerate(raw):
        try:
            probes.append(ProbeEntry(**record))
        except ValidationError as exc:
            raise ValueError(f"Record {i} failed schema validation: {exc}") from exc
    return probes


def validate_probe_bank(probes: list[ProbeEntry]) -> None:
    """Check count, uniqueness, and allowed values; raise ValueError on any failure."""
    if len(probes) != _EXPECTED_TOTAL:
        raise ValueError(f"Expected {_EXPECTED_TOTAL} probes, found {len(probes)}.")

    ids = [p.probe_id for p in probes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate probe_ids detected.")

    invalid_sources = {p.source for p in probes} - _VALID_SOURCES
    if invalid_sources:
        raise ValueError(f"Invalid source values: {invalid_sources}")

    invalid_categories = {p.category for p in probes} - _VALID_CATEGORIES
    if invalid_categories:
        raise ValueError(f"Invalid category values: {invalid_categories}")


def log_summary(probes: list[ProbeEntry]) -> None:
    """Log probe counts broken down by category and source."""
    category_counts = Counter(p.category for p in probes)
    source_counts = Counter(p.source for p in probes)
    logger.info("Probe bank summary — total: %d", len(probes))
    for cat, count in sorted(category_counts.items()):
        logger.info("  category %-28s %d", cat, count)
    for src, count in sorted(source_counts.items()):
        logger.info("  source   %-28s %d", src, count)


def main() -> None:
    """Entry point: load, validate and log the probe bank."""
    config = _load_config()
    path = Path(config["paths"]["probe_bank"])
    probes = load_probe_bank(path)
    validate_probe_bank(probes)
    log_summary(probes)
    logger.info("Stage 1 complete: probe bank is valid.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
