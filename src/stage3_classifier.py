"""Stage 3 — RoBERTa classifier.

Fine-tunes on StereoSet (--mode train) or classifies completions (--mode inference).
"""

from __future__ import annotations

import argparse
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
    """Load dev.json, filter to gender rows, and map labels to binary ints."""
    path = Path(raw_data_dir) / "dev.json"
    with open(path) as f:
        raw = json.load(f)

    df = pd.DataFrame(
        [
            {
                "id": sent["id"],
                "text": sent["sentence"],
                "label": 1 if sent["gold_label"] == "stereotype" else 0,
                "context": item["context"],
            }
            for item in raw["data"]["intersentence"]
            for sent in item["sentences"]
            if item["bias_type"] == "gender" and sent["gold_label"] != "unrelated"
        ]
    )

    assert df["id"].nunique() == len(
        df
    ), "Duplicate sentence IDs found in StereoSet data"
    logger.info(
        "Loaded %d rows from StereoSet (gender, stereotype/anti-stereotype only)",
        len(df),
    )
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
        evaluation_strategy=clf["evaluation_strategy"],
        save_strategy=clf["evaluation_strategy"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        save_total_limit=1,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    logger.info("Model saved to %s", output_dir)

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
        raise NotImplementedError("inference mode not yet implemented")


if __name__ == "__main__":
    main()
