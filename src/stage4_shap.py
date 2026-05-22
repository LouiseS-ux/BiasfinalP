"""
Stage 4 — SHAP Explainer.

Responsibility: Applies SHAP to the fine-tuned classifier to produce
token-level attribution scores. Reads completions.json and saved model.
Writes shap_values.json. Positive SHAP = token pushed toward 'biased'.

Run on biased predictions first. Cap at max_seq_length. Expect 2-5s
per completion on CPU.

Done when: shap_values.json present for all biased predictions.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for Stage 4."""
    raise NotImplementedError("Stage 4 not yet implemented.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
