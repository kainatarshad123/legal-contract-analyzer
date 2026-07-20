"""Validate transformer training and frozen evaluation datasets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


REQUIRED_TRAIN_COLUMNS = {
    "clause_text",
    "clause_type",
    "source_split",
}
REQUIRED_EVAL_COLUMNS = {
    "clause_text",
    "clause_type",
}


def normalize_text(value: object) -> str:
    """Normalize text for exact-overlap checks."""

    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def load_csv(path: Path) -> pd.DataFrame:
    """Load one UTF-8-compatible CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path, encoding="utf-8-sig")


def validate_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    dataset_name: str,
) -> None:
    """Raise when required columns are missing."""

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {sorted(missing)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("transformer_training/data_inspection.json"),
    )
    args = parser.parse_args()

    train_df = load_csv(args.train_csv)
    eval_df = load_csv(args.eval_csv)

    validate_columns(
        train_df,
        REQUIRED_TRAIN_COLUMNS,
        "Training dataset",
    )
    validate_columns(
        eval_df,
        REQUIRED_EVAL_COLUMNS,
        "Frozen evaluation dataset",
    )

    for dataframe in (train_df, eval_df):
        dataframe["clause_text"] = (
            dataframe["clause_text"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        dataframe["clause_type"] = (
            dataframe["clause_type"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    train_df["source_split"] = (
        train_df["source_split"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    train_duplicates = int(
        train_df.duplicated(
            subset=["clause_text", "clause_type"]
        ).sum()
    )
    eval_duplicates = int(
        eval_df.duplicated(
            subset=["clause_text", "clause_type"]
        ).sum()
    )

    train_texts = {
        normalize_text(text)
        for text in train_df["clause_text"]
        if normalize_text(text)
    }
    eval_texts = {
        normalize_text(text)
        for text in eval_df["clause_text"]
        if normalize_text(text)
    }
    overlap_count = len(train_texts & eval_texts)

    summary = {
        "training_rows": int(len(train_df)),
        "evaluation_rows": int(len(eval_df)),
        "training_columns": list(train_df.columns),
        "evaluation_columns": list(eval_df.columns),
        "training_label_count": int(
            train_df["clause_type"].nunique()
        ),
        "evaluation_label_count": int(
            eval_df["clause_type"].nunique()
        ),
        "training_splits": {
            str(key): int(value)
            for key, value in (
                train_df["source_split"]
                .value_counts(dropna=False)
                .to_dict()
                .items()
            )
        },
        "training_duplicates": train_duplicates,
        "evaluation_duplicates": eval_duplicates,
        "normalized_text_overlap": overlap_count,
        "training_label_counts": {
            str(key): int(value)
            for key, value in (
                train_df["clause_type"]
                .value_counts()
                .sort_index()
                .to_dict()
                .items()
            )
        },
        "evaluation_label_counts": {
            str(key): int(value)
            for key, value in (
                eval_df["clause_type"]
                .value_counts()
                .sort_index()
                .to_dict()
                .items()
            )
        },
    }

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output_json.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))

    if train_duplicates:
        raise ValueError(
            f"Training dataset contains {train_duplicates} duplicates."
        )

    if eval_duplicates:
        raise ValueError(
            f"Evaluation dataset contains {eval_duplicates} duplicates."
        )

    if overlap_count:
        raise ValueError(
            "Frozen evaluation text overlaps the training dataset."
        )

    expected_splits = {"train", "validation", "test"}
    available_splits = set(
        train_df["source_split"].unique()
    )
    missing_splits = expected_splits - available_splits

    if missing_splits:
        raise ValueError(
            f"Missing required splits: {sorted(missing_splits)}"
        )

    print("\nDataset validation passed.")


if __name__ == "__main__":
    main()
