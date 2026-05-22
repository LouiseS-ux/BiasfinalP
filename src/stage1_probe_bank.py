"""
Stage 1 — Probe Bank Builder.

Responsibility: Builds gender-paired prompt dataset from WinoBias source
files in data/raw/ plus a set of original hand-written probes. Writes
data/probe_bank.json. Each probe exists in a male and female variant so
divergence in LLM responses can be measured. Assigns stable probe_id
values (wb_001, orig_001).

Done when: probe_bank.json has >= 200 rows, no duplicate probe_ids,
all fields present on every record.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for Stage 1."""
    raise NotImplementedError("Stage 1 not yet implemented.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
