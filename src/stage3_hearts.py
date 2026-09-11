"""Stage 3 HEARTS — ALBERT-v2 bias classifier inference.

Loads the holistic-ai/bias_classifier_albertv2 model from HuggingFace and
runs inference on all completions in data/completions.json. Writes results
to data/hearts_predictions.json and a summary to data/hearts_summary.json.
Keeps all outputs separate from the existing RoBERTa legacy pipeline outputs,
 as this is the replacement final classifier.

LABEL_1 = stereotype (biased), LABEL_0 = not_stereotype (not_biased).
Label mapping confirmed from HEARTS Logistic_Regression.py source code.
"""

from __future__ import annotations

import datetime
import json
import logging
from collections import Counter
from typing import Any

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

HEARTS_MODEL = "holistic-ai/bias_classifier_albertv2"
HEARTS_CACHE_DIR = "models/hearts_albertv2/"


def _load_config() -> dict[str, Any]:
    """Load config.yaml and return it as a dictionary."""
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _load_model_and_tokenizer() -> (
    tuple[AutoModelForSequenceClassification, AutoTokenizer]
):
    """Download and load the HEARTS ALBERT-v2 model and tokenizer."""
    logger.info("Loading HEARTS ALBERT-v2 from %s", HEARTS_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(HEARTS_MODEL, cache_dir=HEARTS_CACHE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        HEARTS_MODEL, cache_dir=HEARTS_CACHE_DIR
    )
    model.eval()
    logger.info("HEARTS model loaded — num_labels: %d", model.config.num_labels)
    logger.info("id2label: %s", model.config.id2label)
    return model, tokenizer


def _classify(
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> tuple[str, float]:
    """Classify a single completion and return label and confidence score."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1).squeeze()
    label_idx = int(torch.argmax(probs))
    label = "biased" if label_idx == 1 else "not_biased"
    confidence = float(probs[label_idx])
    return label, confidence


def _write_summary(
    predictions: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Aggregate HEARTS predictions by model and category and write summary."""
    biased = [p for p in predictions if p["label"] == "biased"]
    by_model: dict[str, Any] = {}

    models = sorted({p["model"] for p in predictions})
    for model_name in models:
        model_preds = [p for p in predictions if p["model"] == model_name]
        model_biased = [p for p in model_preds if p["label"] == "biased"]
        by_category: dict[str, Any] = {}

        cats = sorted({p["category"] for p in model_preds})
        for cat in cats:
            cat_preds = [p for p in model_preds if p["category"] == cat]
            cat_biased = [p for p in cat_preds if p["label"] == "biased"]
            cat_biased_confs = [p["confidence"] for p in cat_biased]
            by_category[cat] = {
                "total": len(cat_preds),
                "biased_count": len(cat_biased),
                "bias_pct": round(100 * len(cat_biased) / len(cat_preds), 2),
                "mean_confidence_biased": (
                    round(sum(cat_biased_confs) / len(cat_biased_confs), 4)
                    if cat_biased_confs
                    else 0.0
                ),
            }

        biased_confs = [p["confidence"] for p in model_biased]
        by_model[model_name] = {
            "total": len(model_preds),
            "biased_count": len(model_biased),
            "bias_pct": round(100 * len(model_biased) / len(model_preds), 2),
            "mean_confidence_biased": (
                round(sum(biased_confs) / len(biased_confs), 4) if biased_confs else 0.0
            ),
            "by_category": by_category,
        }

    summary = {
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "classifier": HEARTS_MODEL,
        "total_completions": len(predictions),
        "total_biased": len(biased),
        "overall_bias_pct": round(100 * len(biased) / len(predictions), 2),
        "by_model": by_model,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s", output_path)


def run_hearts_inference(config: dict[str, Any]) -> None:
    """Run HEARTS classifier on all completions and write predictions."""
    paths = config["paths"]
    clf = config["classifier"]
    max_length: int = clf["max_seq_length"]

    model, tokenizer = _load_model_and_tokenizer()

    with open(paths["completions"]) as f:
        completions = json.load(f)

    with open(paths["probe_bank"]) as f:
        categories = {p["probe_id"]: p["category"] for p in json.load(f)}

    logger.info("Running HEARTS inference on %d completions", len(completions))

    predictions = []
    for i, record in enumerate(completions):
        label, confidence = _classify(
            record["completion"], model, tokenizer, max_length
        )
        predictions.append(
            {
                "probe_id": record["probe_id"],
                "gender": record["gender"],
                "model": record["model"],
                "category": categories.get(record["probe_id"], "unknown"),
                "label": label,
                "confidence": round(confidence, 4),
                "completion": record["completion"],
            }
        )
        if (i + 1) % 100 == 0:
            logger.info("Processed %d / %d completions", i + 1, len(completions))

    output_path = "data/hearts_predictions.json"
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info("HEARTS predictions written to %s", output_path)

    _write_summary(predictions, "data/hearts_summary.json")

    biased = [p for p in predictions if p["label"] == "biased"]
    biased_confs = [p["confidence"] for p in biased]
    logger.info(
        "Total biased: %d / %d (%.1f%%)",
        len(biased),
        len(predictions),
        100 * len(biased) / len(predictions),
    )
    logger.info(
        "Mean confidence (biased only): %.4f",
        sum(biased_confs) / len(biased_confs) if biased_confs else 0.0,
    )

    cat_counts = Counter(p["category"] for p in biased)
    for cat, count in sorted(cat_counts.items()):
        logger.info("  %s: %d biased", cat, count)


def main() -> None:
    """Entry point — load config and run HEARTS inference."""
    logging.basicConfig(level=logging.INFO)
    config = _load_config()
    run_hearts_inference(config)


if __name__ == "__main__":
    main()
