"""Stage 4 — SHAP explainer.

Loads the fine-tuned RoBERTa classifier and computes token-level SHAP attribution
scores for all 600 completions. Writes data/shap_values.json incrementally and
runs six post-hoc evaluation checks on the classifier's behaviour.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import scipy.special
import shap
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# Gendered pronouns used in pronoun masking ablation (check 2)
_PRONOUNS = {
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "He",
    "Him",
    "His",
    "She",
    "Her",
    "Hers",
}


def _load_config() -> dict[str, Any]:
    """Load config.yaml and return it as a dictionary."""
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _load_model_and_tokenizer(
    output_dir: str,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load the fine-tuned classifier and tokenizer from the checkpoint directory."""
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.eval()
    return model, tokenizer


def _make_predict_fn(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Callable[..., np.ndarray]:
    """Return a prediction function that outputs logit scores for the biased class."""

    def predict(texts: list[str]) -> np.ndarray:
        """Score a batch of texts, returning logit for the biased class (label 1)."""
        inputs = tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        with torch.no_grad():
            logits = model(**inputs).logits.numpy()
        probs = scipy.special.softmax(logits, axis=1)
        return scipy.special.logit(probs[:, 1])

    return predict


def _keep_alive(interval_seconds: int = 1200) -> None:
    """Log a keep-alive ping every interval_seconds to prevent session timeout."""
    while True:
        time.sleep(interval_seconds)
        logger.info("Keep-alive ping — SHAP still running")


def _load_existing_shap(
    shap_path: str,
) -> tuple[list[dict[str, Any]], set[tuple[str, str, str]]]:
    """Load existing shap_values.json and return records plus any completed keys."""
    path = Path(shap_path)
    if not path.exists():
        return [], set()
    with open(shap_path) as f:
        records = json.load(f)
    completed = {(r["probe_id"], r["gender"], r["model"]) for r in records}
    logger.info("Resuming — %d completions already processed", len(records))
    return records, completed


def _append_shap_record(
    shap_path: str,
    records: list[dict[str, Any]],
    new_record: dict[str, Any],
) -> None:
    """Append a new SHAP record to the in-memory list and write all to disk."""
    records.append(new_record)
    with open(shap_path, "w") as f:
        json.dump(records, f, indent=2)


def _check1_shap_mass(shap_records: list[dict[str, Any]]) -> None:
    """Check 1: compute pronoun vs non-pronoun SHAP mass distribution."""
    pronoun_mass = []
    total_mass = []
    for rec in shap_records:
        tokens = rec["tokens"]
        values = rec["shap"]
        pos_values = [max(v, 0.0) for v in values]
        total = sum(pos_values)
        pronoun = sum(
            v
            for t, v in zip(tokens, pos_values, strict=False)
            if t.strip() in _PRONOUNS
        )
        if total > 0:
            pronoun_mass.append(pronoun / total)
            total_mass.append(total)

    avg_pronoun_pct = float(np.mean(pronoun_mass)) * 100 if pronoun_mass else 0.0
    logger.info(
        "Check 1 — Pronoun SHAP mass: %.1f%% of positive attribution on average",
        avg_pronoun_pct,
    )


def _check2_pronoun_masking(
    completions: list[dict[str, Any]],
    predict_fn: Callable[..., np.ndarray],
) -> None:
    """Check 2: mask gendered pronouns with [PERSON] and compare labels."""
    flipped = 0
    total = 0
    for record in completions:
        original = record["completion"]
        masked = re.sub(
            r"\b(" + "|".join(_PRONOUNS) + r")\b",
            "[PERSON]",
            original,
        )
        orig_score = float(predict_fn([original])[0])
        mask_score = float(predict_fn([masked])[0])
        orig_label = "biased" if orig_score > 0 else "not_biased"
        mask_label = "biased" if mask_score > 0 else "not_biased"
        if orig_label != mask_label:
            flipped += 1
        total += 1

    logger.info(
        "Check 2 — Pronoun masking: %d/%d completions flipped label after masking"
        " (%.1f%%)",
        flipped,
        total,
        100 * flipped / total if total > 0 else 0.0,
    )


def _check3_paired_shap(shap_records: list[dict[str, Any]]) -> None:
    """Check 3: compare male vs female SHAP top tokens per probe."""
    by_probe: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for rec in shap_records:
        pid = rec["probe_id"]
        gender = rec["gender"]
        if pid not in by_probe:
            by_probe[pid] = {}
        top = sorted(
            zip(rec["tokens"], rec["shap"], strict=False),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]
        by_probe[pid][gender] = top

    asymmetric = 0
    for _, genders in by_probe.items():
        if "male" in genders and "female" in genders:
            male_tokens = {t.strip() for t, _ in genders["male"]}
            female_tokens = {t.strip() for t, _ in genders["female"]}
            if male_tokens != female_tokens:
                asymmetric += 1

    logger.info(
        "Check 3 — Paired SHAP: %d/%d probe pairs show asymmetric top tokens",
        asymmetric,
        len(by_probe),
    )


def _check4_nonpronoun_tokens(shap_records: list[dict[str, Any]]) -> None:
    """Check 4: rank non-pronoun tokens by mean positive SHAP attribution."""
    token_scores: dict[str, list[float]] = {}
    for rec in shap_records:
        for token, value in zip(rec["tokens"], rec["shap"], strict=False):
            clean = token.strip()
            if clean in _PRONOUNS or not clean or clean in {".", ",", "!", "?"}:
                continue
            if value > 0:
                token_scores.setdefault(clean, []).append(value)

    ranked = sorted(
        ((t, float(np.mean(v))) for t, v in token_scores.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    logger.info("Check 4 — Top non-pronoun bias tokens:")
    for token, score in ranked:
        logger.info("  %-20s %.4f", token, score)


def _check5_model_comparison(shap_records: list[dict[str, Any]]) -> None:
    """Check 5: compare top SHAP tokens between GPT-4o and Claude per probe."""
    by_probe: dict[str, dict[str, set[str]]] = {}
    for rec in shap_records:
        pid = rec["probe_id"]
        model = rec["model"]
        if pid not in by_probe:
            by_probe[pid] = {}
        top = {
            t.strip()
            for t, v in sorted(
                zip(rec["tokens"], rec["shap"], strict=False),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]
        }
        by_probe[pid][model] = top

    shared = 0
    divergent = 0
    for _, models in by_probe.items():
        if len(models) == 2:
            tokens_list = list(models.values())
            overlap = tokens_list[0] & tokens_list[1]
            if len(overlap) >= 3:
                shared += 1
            else:
                divergent += 1

    logger.info(
        "Check 5 — Model comparison: %d probes share top bias tokens across"
        " models, %d diverge",
        shared,
        divergent,
    )


def _check6_category_attribution(
    shap_records: list[dict[str, Any]],
    completions: list[dict[str, Any]],
) -> None:
    """Check 6: average positive SHAP attribution by probe category."""
    probe_categories: dict[str, str] = {}
    for rec in completions:
        if "category" in rec:
            probe_categories[rec["probe_id"]] = rec["category"]

    category_scores: dict[str, list[float]] = {}
    for rec in shap_records:
        category = probe_categories.get(rec["probe_id"], "unknown")
        total_pos = sum(max(v, 0.0) for v in rec["shap"])
        category_scores.setdefault(category, []).append(total_pos)

    logger.info("Check 6 — Mean positive SHAP attribution by category:")
    for cat, scores in sorted(category_scores.items()):
        logger.info("  %-30s %.4f", cat, float(np.mean(scores)))


def run_shap(config: dict[str, Any]) -> None:
    """Run SHAP on all completions incrementally and write shap_values.json."""
    paths = config["paths"]
    clf = config["classifier"]
    max_length: int = clf["max_seq_length"]
    output_dir: str = clf["output_dir"]
    max_evals: int = clf.get("shap_max_evals", 200)

    # Start keep-alive thread to prevent session timeout
    keepalive = threading.Thread(target=_keep_alive, daemon=True)
    keepalive.start()

    model, tokenizer = _load_model_and_tokenizer(output_dir)
    predict_fn = _make_predict_fn(model, tokenizer, max_length)
    explainer = shap.Explainer(predict_fn, tokenizer)

    with open(paths["completions"]) as f:
        completions = json.load(f)

    with open(paths["probe_bank"]) as f:
        categories = {p["probe_id"]: p["category"] for p in json.load(f)}

    for record in completions:
        record["category"] = categories.get(record["probe_id"], "unknown")

    Path(paths["shap_values"]).parent.mkdir(parents=True, exist_ok=True)
    shap_records, completed = _load_existing_shap(paths["shap_values"])

    remaining = [
        r
        for r in completions
        if (r["probe_id"], r["gender"], r["model"]) not in completed
    ]

    logger.info(
        "Running SHAP on %d completions (%d remaining) with max_evals=%d",
        len(completions),
        len(remaining),
        max_evals,
    )

    for i, record in enumerate(remaining):
        shap_vals = explainer(
            [record["completion"]],
            fixed_context=1,
            max_evals=max_evals,
        )
        tokens = shap_vals[0].data.tolist()
        values = shap_vals[0].values.tolist()
        new_record = {
            "probe_id": record["probe_id"],
            "gender": record["gender"],
            "model": record["model"],
            "tokens": tokens,
            "shap": [round(v, 6) for v in values],
        }
        _append_shap_record(paths["shap_values"], shap_records, new_record)
        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d completions processed", i + 1, len(remaining))

    logger.info("SHAP values written to %s", paths["shap_values"])
    logger.info("Running post-hoc evaluation checks...")
    _check1_shap_mass(shap_records)
    _check2_pronoun_masking(completions, predict_fn)
    _check3_paired_shap(shap_records)
    _check4_nonpronoun_tokens(shap_records)
    _check5_model_comparison(shap_records)
    _check6_category_attribution(shap_records, completions)


def main() -> None:
    """Load config and run SHAP explainer over all completions."""
    logging.basicConfig(level=logging.INFO)
    config = _load_config()
    run_shap(config)


if __name__ == "__main__":
    main()
