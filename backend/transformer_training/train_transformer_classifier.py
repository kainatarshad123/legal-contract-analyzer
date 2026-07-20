"""Fine-tune DistilBERT for 22-class legal clause classification."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def set_seed(seed: int) -> None:
    """Set reproducible random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path: Path) -> Dataset:
    """Load a prepared CSV into a Hugging Face Dataset."""

    dataframe = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    required = {
        "clause_text",
        "label_id",
    }
    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{path} missing columns: {sorted(missing)}"
        )

    dataframe = dataframe[
        ["clause_text", "label_id"]
    ].copy()
    dataframe["label_id"] = (
        dataframe["label_id"].astype(int)
    )

    return Dataset.from_pandas(
        dataframe,
        preserve_index=False,
    )


def build_compute_metrics() -> Any:
    """Return a Trainer metrics callback."""

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy_score(
                labels,
                predictions,
            ),
            "macro_f1": f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            ),
        }

    return compute_metrics


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weights = (
            self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )

        loss_function = torch.nn.CrossEntropyLoss(
            weight=weights
        )
        loss = loss_function(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("transformer_training/prepared_data"),
    )
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "ml_model/transformer_clause_model"
        ),
    )
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    mapping_path = (
        args.prepared_dir / "label_mapping.json"
    )
    mapping = json.loads(
        mapping_path.read_text(encoding="utf-8")
    )

    label2id = {
        str(label): int(index)
        for label, index in mapping["label2id"].items()
    }
    id2label = {
        int(index): str(label)
        for index, label in mapping["id2label"].items()
    }

    train_dataset = load_split(
        args.prepared_dir / "train.csv"
    )
    validation_dataset = load_split(
        args.prepared_dir / "validation.csv"
    )
    test_dataset = load_split(
        args.prepared_dir / "test.csv"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )

    def tokenize_batch(
        batch: dict[str, list[Any]],
    ) -> dict[str, Any]:
        return tokenizer(
            batch["clause_text"],
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["clause_text"],
    )
    validation_dataset = validation_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["clause_text"],
    )
    test_dataset = test_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["clause_text"],
    )

    train_dataset = train_dataset.rename_column(
        "label_id",
        "labels",
    )
    validation_dataset = validation_dataset.rename_column(
        "label_id",
        "labels",
    )
    test_dataset = test_dataset.rename_column(
        "label_id",
        "labels",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    class_weights = None

    if not args.no_class_weights:
        train_labels = np.array(
            train_dataset["labels"],
            dtype=int,
        )
        classes = np.arange(len(label2id))
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=train_labels,
        )
        class_weights = torch.tensor(
            weights,
            dtype=torch.float32,
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(
        use_cuda
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = bool(use_cuda and not use_bf16)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation
        ),
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=0,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
        ),
        compute_metrics=build_compute_metrics(),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
            )
        ],
        class_weights=class_weights,
    )

    train_result = trainer.train()
    validation_metrics = trainer.evaluate(
        validation_dataset,
        metric_key_prefix="validation",
    )
    test_metrics = trainer.evaluate(
        test_dataset,
        metric_key_prefix="test",
    )

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    mapping_copy = args.output_dir / "label_mapping.json"
    mapping_copy.write_text(
        json.dumps(mapping, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "base_model": args.model_name,
        "max_length": args.max_length,
        "class_weights_enabled": (
            not args.no_class_weights
        ),
        "device": (
            "cuda"
            if use_cuda
            else "cpu"
        ),
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    metrics_path = (
        args.output_dir / "training_metrics.json"
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("Transformer training completed.")
    print(f"Model saved to: {args.output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
