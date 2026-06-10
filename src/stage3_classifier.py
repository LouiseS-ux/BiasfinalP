"""
Stage 3 — RoBERTa Classifier.

Objective - Fine-tunes classifier head on roberta-base model on StereoSet labelled data.
Runs inference over all completions. Two modes via --mode flag:
  --mode train     Fine-tune and save best checkpoint by eval F1.
  --mode inference Load checkpoint, write predictions.json and
                   summary_stats.json.

Target F1 >= 0.75. All training config in config.yaml under classifier: key.

Done when: F1 >= 0.75; checkpoint saved; predictions.json written.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for Stage 3."""
    raise NotImplementedError("Stage 3 not yet implemented.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
