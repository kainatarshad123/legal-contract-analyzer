"""Evaluate a fine-tuned transformer on the frozen 330-clause set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def batched(values: list[str], size: int) -> list[list[str]]:
    """Yield list batches."""

    return [
        values[index:index + size]
        for index in range(0, len(values), size)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "ml_model/transformer_clause_model"
        ),
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=Path(
            "transformer_training/prepared_data/"
            "frozen_external_eval.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "transformer_training/results"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()

    dataframe = pd.read_csv(
        args.eval_csv,
        encoding="utf-8-sig",
    )

    mapping = json.loads(
        (
            args.model_dir / "label_mapping.json"
        ).read_text(encoding="utf-8")
    )
    labels = list(mapping["labels"])
    label2id = {
        str(label): int(index)
        for label, index in mapping["label2id"].items()
    }
    id2label = {
        int(index): str(label)
        for index, label in mapping["id2label"].items()
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir
    )
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(args.model_dir)
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)
    model.eval()

    texts = (
        dataframe["clause_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )
    true_labels = (
        dataframe["clause_type"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    predicted_ids: list[int] = []
    confidence_scores: list[float] = []

    start_time = time.perf_counter()

    with torch.inference_mode():
        for text_batch in batched(
            texts,
            args.batch_size,
        ):
            encoded = tokenizer(
                text_batch,
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            logits = model(**encoded).logits
            probabilities = torch.softmax(
                logits,
                dim=-1,
            )
            confidence, prediction = (
                probabilities.max(dim=-1)
            )

            predicted_ids.extend(
                prediction.cpu().tolist()
            )
            confidence_scores.extend(
                confidence.cpu().tolist()
            )

    elapsed = time.perf_counter() - start_time
    predicted_labels = [
        id2label[index]
        for index in predicted_ids
    ]

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )
    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    report_dict = classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .reset_index()
        .rename(columns={"index": "label"})
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=labels,
    )
    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    predictions_df = dataframe.copy()
    predictions_df["transformer_prediction"] = (
        predicted_labels
    )
    predictions_df["transformer_confidence"] = (
        confidence_scores
    )
    predictions_df["transformer_correct"] = (
        predictions_df["clause_type"]
        == predictions_df["transformer_prediction"]
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_csv(
        args.output_dir
        / "transformer_external_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report_df.to_csv(
        args.output_dir
        / "transformer_classification_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    matrix_df.to_csv(
        args.output_dir
        / "transformer_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    model_size_bytes = sum(
        path.stat().st_size
        for path in args.model_dir.rglob("*")
        if path.is_file()
    )

    summary = {
        "evaluation_rows": len(dataframe),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "average_confidence": float(
            np.mean(confidence_scores)
        ),
        "elapsed_seconds": elapsed,
        "milliseconds_per_clause": (
            elapsed / max(1, len(dataframe)) * 1000
        ),
        "device": str(device),
        "model_size_bytes": model_size_bytes,
        "model_size_megabytes": (
            model_size_bytes / (1024 * 1024)
        ),
        "labels_in_training": len(labels),
        "labels_with_eval_support": int(
            dataframe["clause_type"].nunique()
        ),
    }

    (
        args.output_dir
        / "transformer_external_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
