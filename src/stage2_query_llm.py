"""
Stage 2 — LLM Query Runner.

Responsibility: Feeds each probe to each configured LLM (gpt-4o and
claude-opus-4-5) and collects completions into data/completions.json.
Resume-safe — skips probe/model pairs already collected. Uses configurable
delay (default 0.5s from config.yaml). Implements exponential backoff on
429 errors. Logs token usage per call for cost tracking.

Done when: >= 600 completions; all probe x gender x model combos present.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from src.adapters.anthropic_adapter import AnthropicAdapter
from src.adapters.openai_adapter import OpenAIAdapter
from src.llm_client import LLMClient, LLMError

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path("config.yaml")
_GENDERS: tuple[str, str] = ("male", "female")


def _load_config() -> dict[str, Any]:
    """Load config.yaml and return it as a dictionary."""
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_probe_bank(path: Path) -> list[dict[str, Any]]:
    """Load probe_bank.json and return the raw list of probe dicts."""
    with open(path) as f:
        return json.load(f)


def _load_existing_completions(path: Path) -> list[dict[str, Any]]:
    """Return existing completions from disk, or an empty list if the file is absent."""
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _seen_keys(completions: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    """Return (probe_id, gender, model) triples already present in completions."""
    return {(c["probe_id"], c["gender"], c["model"]) for c in completions}


def _write_completions(path: Path, completions: list[dict[str, Any]]) -> None:
    """Write the completions list to disk as indented JSON."""
    path.write_text(json.dumps(completions, indent=2))


def _build_clients(config: dict[str, Any]) -> dict[str, LLMClient]:
    """Instantiate one LLMClient per configured model, keyed by model name."""
    return {
        config["api"]["models"]["openai"]: OpenAIAdapter(config),
        config["api"]["models"]["anthropic"]: AnthropicAdapter(config),
    }


def run_queries(
    probes: list[dict[str, Any]],
    clients: dict[str, LLMClient],
    completions: list[dict[str, Any]],
    completions_path: Path,
    delay: float,
) -> tuple[int, int, int]:
    """Run all probe × gender × model queries, skip seen combos, write after each."""
    seen = _seen_keys(completions)
    collected = 0
    skipped = 0
    failures = 0

    for probe in probes:
        probe_id: str = probe["probe_id"]
        for gender in _GENDERS:
            prompt: str = probe[f"{gender}_prompt"]
            for model_name, client in clients.items():
                if (probe_id, gender, model_name) in seen:
                    skipped += 1
                    continue
                try:
                    record = client.complete(
                        prompt=prompt,
                        probe_id=probe_id,
                        gender=gender,
                    )
                    completions.append(record.model_dump())
                    seen.add((probe_id, gender, model_name))
                    _write_completions(completions_path, completions)
                    collected += 1
                except LLMError as exc:
                    logger.error(
                        "LLMError probe=%s gender=%s model=%s: %s",
                        probe_id,
                        gender,
                        model_name,
                        exc,
                    )
                    failures += 1
                time.sleep(delay)

    return collected, skipped, failures


def main() -> None:
    """Load config and probe bank, query all models, and log a completion summary."""
    config = _load_config()
    probe_bank_path = Path(config["paths"]["probe_bank"])
    completions_path = Path(config["paths"]["completions"])
    delay: float = config["api"]["call_delay_seconds"]

    probes = _load_probe_bank(probe_bank_path)
    completions = _load_existing_completions(completions_path)
    clients = _build_clients(config)

    logger.info(
        "Stage 2: %d probes, %d existing completions", len(probes), len(completions)
    )

    collected, skipped, failures = run_queries(
        probes=probes,
        clients=clients,
        completions=completions,
        completions_path=completions_path,
        delay=delay,
    )

    logger.info(
        "Stage 2 complete — collected: %d  skipped: %d  failures: %d",
        collected,
        skipped,
        failures,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
