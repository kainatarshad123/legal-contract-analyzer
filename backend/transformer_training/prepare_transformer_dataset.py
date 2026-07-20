"""Prepare normalized CSV files and label mappings for transformer training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean required text and label columns."""

    cleaned = dataframe.copy()
    cleaned["clause_text"] = (
        cleaned["clause_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    cleaned["clause_type"] = (
        cleaned["clause_type"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cleaned = cleaned[
        (cleaned["clause_text"].str.len() > 0)
        & (cleaned["clause_type"].str.len() > 0)
    ].copy()

    return cleaned.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transformer_training/prepared_data"),
    )
    args = parser.parse_args()

    train_df = pd.read_csv(
        args.train_csv,
        encoding="utf-8-sig",
    )
    eval_df = pd.read_csv(
        args.eval_csv,
        encoding="utf-8-sig",
    )

    required_train = {
        "clause_text",
        "clause_type",
        "source_split",
    }
    required_eval = {
        "clause_text",
        "clause_type",
    }

    missing_train = required_train - set(train_df.columns)
    missing_eval = required_eval - set(eval_df.columns)

    if missing_train:
        raise ValueError(
            f"Training CSV missing columns: {sorted(missing_train)}"
        )

    if missing_eval:
        raise ValueError(
            f"Evaluation CSV missing columns: {sorted(missing_eval)}"
        )

    train_df = clean_dataframe(train_df)
    eval_df = clean_dataframe(eval_df)

    train_df["source_split"] = (
        train_df["source_split"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    labels = sorted(train_df["clause_type"].unique())
    label2id = {
        label: index
        for index, label in enumerate(labels)
    }
    id2label = {
        str(index): label
        for label, index in label2id.items()
    }

    unknown_eval_labels = sorted(
        set(eval_df["clause_type"]) - set(labels)
    )

    if unknown_eval_labels:
        raise ValueError(
            "Frozen evaluation set contains unseen labels: "
            f"{unknown_eval_labels}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_files: dict[str, str] = {}

    for split in ("train", "validation", "test"):
        split_df = train_df[
            train_df["source_split"] == split
        ].copy()

        if split_df.empty:
            raise ValueError(
                f"Prepared {split} split is empty."
            )

        split_df["label_id"] = (
            split_df["clause_type"]
            .map(label2id)
            .astype(int)
        )

        output_path = args.output_dir / f"{split}.csv"
        split_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )
        split_files[split] = str(output_path)

    eval_df["label_id"] = (
        eval_df["clause_type"]
        .map(label2id)
        .astype(int)
    )
    eval_output = args.output_dir / "frozen_external_eval.csv"
    eval_df.to_csv(
        eval_output,
        index=False,
        encoding="utf-8-sig",
    )

    mapping = {
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
        "num_labels": len(labels),
        "split_files": split_files,
        "frozen_external_eval": str(eval_output),
    }

    mapping_path = args.output_dir / "label_mapping.json"
    mapping_path.write_text(
        json.dumps(mapping, indent=2),
        encoding="utf-8",
    )

    print("Transformer data prepared successfully.")
    print(f"Labels: {len(labels)}")
    print(f"Mapping: {mapping_path}")
    print(f"Frozen evaluation: {eval_output}")


if __name__ == "__main__":
    main()
