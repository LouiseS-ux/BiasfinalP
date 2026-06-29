"""Stage 3 — RoBERTa classifier.

Fine-tunes on StereoSet (--mode train) or classifies completions (--mode inference).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def _load_config() -> dict[str, Any]:
    """Load config.yaml and return it as a dictionary."""
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_stereoset(raw_data_dir: str) -> pd.DataFrame:
    """Load dev.json, filter gender rows from both splits, return labelled DataFrame."""
    path = Path(raw_data_dir) / "dev.json"
    with open(path) as f:
        raw = json.load(f)

    rows = []
    for split in ["intersentence", "intrasentence"]:
        for item in raw["data"][split]:
            if item["bias_type"] != "gender":
                continue
            for sent in item["sentences"]:
                if sent["gold_label"] == "unrelated":
                    continue
                rows.append(
                    {
                        "id": f"{split}_{sent['id']}",
                        "text": sent["sentence"],
                        "label": 1 if sent["gold_label"] == "stereotype" else 0,
                        "context": item["context"],
                        "split": split,
                    }
                )

    df = pd.DataFrame(rows)
    assert df["id"].nunique() == len(
        df
    ), "Duplicate sentence IDs found in StereoSet data"
    logger.info("Loaded %d rows from StereoSet (gender only)", len(df))
    return df


def _log_label_distribution(name: str, labels: list[int]) -> None:
    """Log the biased/not_biased count for a dataset split."""
    counts = pd.Series(labels).value_counts()
    logger.info(
        "%s label distribution — biased (1): %d, not_biased (0): %d",
        name,
        int(counts.get(1, 0)),
        int(counts.get(0, 0)),
    )


def _compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute precision, recall and F1 from Trainer eval predictions."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _tokenize_dataset(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    """Tokenise a DataFrame with a text column and return a HuggingFace Dataset."""
    dataset = Dataset.from_pandas(df[["text", "label"]])

    def tokenize_fn(examples: dict[str, Any]) -> dict[str, Any]:
        """Tokenise a batch of text examples with padding and truncation."""
        return tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.map(lambda examples: {"labels": examples["label"]})
    return dataset


def _write_metrics(
    metrics_path: str,
    f1: float,
    precision: float,
    recall: float,
    epoch_count: int,
    train_size: int,
    test_size: int,
    label_distribution: dict[str, int],
) -> None:
    """Write training metrics to metrics.json."""
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "epoch_count": epoch_count,
        "train_size": train_size,
        "test_size": test_size,
        "label_distribution": label_distribution,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics written to %s", metrics_path)


def _evaluate_by_split(
    trainer: Trainer,
    test_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> None:
    """Evaluate the model separately on intersentence and intrasentence test rows."""
    for split_name in ["intersentence", "intrasentence"]:
        split_df = test_df[test_df["split"] == split_name].reset_index(drop=True)
        split_dataset = _tokenize_dataset(split_df, tokenizer, max_length)
        split_results = trainer.evaluate(split_dataset)
        logger.info(
            "%s F1: %.4f, Precision: %.4f, Recall: %.4f",
            split_name,
            split_results["eval_f1"],
            split_results["eval_precision"],
            split_results["eval_recall"],
        )


def _write_summary_stats(
    predictions: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Aggregate predictions by model and category and write summary_stats.json."""
    df = pd.DataFrame(predictions)
    summary: dict[str, Any] = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "total_completions": len(df),
        "by_model": {},
    }

    for model_name, model_df in df.groupby("model"):
        biased = model_df[model_df["label"] == "biased"]
        by_category: dict[str, Any] = {}

        if "category" in model_df.columns:
            for cat, cat_df in model_df.groupby("category"):
                cat_biased = (cat_df["label"] == "biased").sum()
                by_category[cat] = {
                    "bias_pct": round(100 * cat_biased / len(cat_df), 2)
                }

        summary["by_model"][model_name] = {
            "total": len(model_df),
            "biased_count": len(biased),
            "bias_pct": round(100 * len(biased) / len(model_df), 2),
            "by_category": by_category,
        }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary stats written to %s", output_path)


def run_train(config: dict[str, Any]) -> None:
    """Fine-tune RoBERTa on StereoSet gender data and save the best checkpoint."""
    clf = config["classifier"]
    seed: int = clf["seed"]
    _set_seeds(seed)

    df = _load_stereoset(config["paths"]["raw_data"])

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info("Train size: %d, Test size: %d", len(train_df), len(test_df))
    _log_label_distribution("Train", train_df["label"].tolist())
    _log_label_distribution("Test", test_df["label"].tolist())

    model_name: str = clf["model_name"]
    max_length: int = clf["max_seq_length"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
    )

    tokenized_train = _tokenize_dataset(train_df, tokenizer, max_length)
    tokenized_test = _tokenize_dataset(test_df, tokenizer, max_length)

    output_dir: str = clf["output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=clf["num_epochs"],
        per_device_train_batch_size=clf["batch_size"],
        per_device_eval_batch_size=clf["batch_size"],
        learning_rate=clf["learning_rate"],
        weight_decay=clf["weight_decay"],
        eval_strategy=clf["eval_strategy"],
        save_strategy=clf["eval_strategy"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        save_total_limit=1,
        save_only_model=True,
        seed=seed,
        warmup_steps=clf["warmup_steps"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    logger.info("Model saved to %s", output_dir)

    _evaluate_by_split(trainer, test_df, tokenizer, max_length)

    eval_results = trainer.evaluate()
    f1 = eval_results.get("eval_f1", 0.0)
    precision = eval_results.get("eval_precision", 0.0)
    recall = eval_results.get("eval_recall", 0.0)

    target_f1: float = clf["target_f1"]
    if f1 >= target_f1:
        logger.info("Target F1 met: %.4f >= %.2f", f1, target_f1)
    else:
        logger.warning("Target F1 NOT met: %.4f < %.2f", f1, target_f1)

    label_counts = df["label"].value_counts()
    _write_metrics(
        metrics_path=config["paths"]["metrics"],
        f1=f1,
        precision=precision,
        recall=recall,
        epoch_count=clf["num_epochs"],
        train_size=len(train_df),
        test_size=len(test_df),
        label_distribution={
            "biased": int(label_counts.get(1, 0)),
            "not_biased": int(label_counts.get(0, 0)),
        },
    )


def run_inference(config: dict[str, Any]) -> None:
    """Load the fine-tuned classifier and classify all completions."""
    clf = config["classifier"]
    paths = config["paths"]

    output_dir: str = clf["output_dir"]
    max_length: int = clf["max_seq_length"]

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.eval()

    with open(paths["completions"]) as f:
        completions = json.load(f)

    logger.info("Loaded %d completions for inference", len(completions))

    predictions = []
    for record in completions:
        inputs = tokenizer(
            record["completion"],
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

        predictions.append(
            {
                "probe_id": record["probe_id"],
                "gender": record["gender"],
                "model": record["model"],
                "label": label,
                "confidence": round(confidence, 4),
                "completion": record["completion"],
            }
        )

    with open(paths["predictions"], "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info("Predictions written to %s", paths["predictions"])

    _write_summary_stats(predictions, paths["summary_stats"])


def main() -> None:
    """Parse --mode argument and dispatch to train or inference."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Stage 3 RoBERTa classifier")
    parser.add_argument(
        "--mode",
        choices=["train", "inference"],
        required=True,
        help="Run mode: train or inference",
    )
    args = parser.parse_args()

    config = _load_config()

    if args.mode == "train":
        run_train(config)
    elif args.mode == "inference":
        run_inference(config)


if __name__ == "__main__":
    main()
