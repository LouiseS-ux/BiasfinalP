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

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for Stage 2."""
    raise NotImplementedError("Stage 2 not yet implemented.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
