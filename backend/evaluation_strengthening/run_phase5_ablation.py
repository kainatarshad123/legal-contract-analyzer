from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND_DIR / "ml_model"
OUTPUT_DIR = BACKEND_DIR / "evaluation"

TRAINING_FILE = MODEL_DIR / "combined_clause_type_dataset_v2.csv"
EVALUATION_FILE = MODEL_DIR / "external_test_clauses_final.csv"

PHASE5_ORIGIN = "Phase 5 Auto-Labeled Lease PDFs"

CLASS_NAMES: list[str] = [
    "Alterations / Improvements",
    "Assignment / Subletting",
    "Confidentiality",
    "Dispute Resolution",
    "General Obligation",
    "Governing Law",
    "Insurance",
    "Lease Grant",
    "Liability / Indemnity",
    "Notice",
    "Other / Unknown",
    "Payment / Rent",
    "Possession / Surrender",
    "Property / Premises",
    "Quiet Enjoyment",
    "Repairs / Maintenance",
    "Signature / Execution",
    "Taxes / Utilities",
    "Term and Renewal",
    "Termination / Default",
    "Use of Premises",
    "Warranties / Representations",
]


def clean_training_data() -> pd.DataFrame:
    dataframe = pd.read_csv(
        TRAINING_FILE,
        encoding="utf-8-sig",
    )

    required_columns = {
        "clause_text",
        "clause_type",
        "source_split",
        "dataset_origin",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Training dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    for column in required_columns:
        dataframe[column] = (
            dataframe[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    dataframe = dataframe[
        (dataframe["clause_text"] != "")
        & (dataframe["clause_type"] != "")
    ].copy()

    return dataframe.reset_index(drop=True)


def clean_evaluation_data() -> pd.DataFrame:
    dataframe = pd.read_csv(
        EVALUATION_FILE,
        encoding="utf-8-sig",
    )

    required_columns = {
        "clause_text",
        "clause_type",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Evaluation dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

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

    if "corrected_clause_type" in dataframe.columns:
        corrected_labels = (
            dataframe["corrected_clause_type"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        dataframe["clause_type"] = corrected_labels.where(
            corrected_labels != "",
            dataframe["clause_type"],
        )

    if "include_in_evaluation" in dataframe.columns:
        include_mask = (
            dataframe["include_in_evaluation"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes")
        )

        dataframe = dataframe[include_mask].copy()

    dataframe = dataframe[
        (dataframe["clause_text"] != "")
        & (dataframe["clause_type"] != "")
    ].copy()

    return dataframe.reset_index(drop=True)


def train_and_evaluate(
    training_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    experiment_name: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )

    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )

    training_features = vectorizer.fit_transform(
        training_data["clause_text"]
    )

    classifier.fit(
        training_features,
        training_data["clause_type"],
    )

    evaluation_features = vectorizer.transform(
        evaluation_data["clause_text"]
    )

    predictions = classifier.predict(
        evaluation_features
    )

    report: dict[str, Any] = classification_report(
        evaluation_data["clause_type"],
        predictions,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_dataframe = pd.DataFrame(
        report
    ).transpose()

    report_dataframe.to_csv(
        OUTPUT_DIR
        / f"{experiment_name}_classification_report.csv",
        encoding="utf-8-sig",
    )

    metrics: dict[str, float | int | str] = {
        "experiment": experiment_name,
        "training_rows": len(training_data),
        "phase5_rows": int(
            (
                training_data["dataset_origin"]
                == PHASE5_ORIGIN
            ).sum()
        ),
        "accuracy": accuracy_score(
            evaluation_data["clause_type"],
            predictions,
        ),
        "macro_f1": f1_score(
            evaluation_data["clause_type"],
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            evaluation_data["clause_type"],
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    return metrics, report_dataframe


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAINING_FILE}"
        )

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Frozen evaluation dataset not found:\n{EVALUATION_FILE}"
        )

    print("Loading training dataset...")

    all_training_data = clean_training_data()

    print("Loading frozen evaluation dataset...")

    evaluation_data = clean_evaluation_data()

    train_only = all_training_data[
        all_training_data["source_split"]
        .str.lower()
        .eq("train")
    ].copy()

    without_phase5 = train_only[
        train_only["dataset_origin"] != PHASE5_ORIGIN
    ].copy()

    with_phase5 = train_only.copy()

    print("\nTraining configuration A:")
    print("Without Phase-5 provisional labels")
    print(f"Training rows: {len(without_phase5)}")

    without_metrics, without_report = train_and_evaluate(
        training_data=without_phase5,
        evaluation_data=evaluation_data,
        experiment_name="ablation_without_phase5",
    )

    print("\nTraining configuration B:")
    print("With Phase-5 provisional labels")
    print(f"Training rows: {len(with_phase5)}")

    with_metrics, with_report = train_and_evaluate(
        training_data=with_phase5,
        evaluation_data=evaluation_data,
        experiment_name="ablation_with_phase5",
    )

    accuracy_difference: float = (
        float(with_metrics["accuracy"])
        - float(without_metrics["accuracy"])
    )

    macro_f1_difference: float = (
        float(with_metrics["macro_f1"])
        - float(without_metrics["macro_f1"])
    )

    weighted_f1_difference: float = (
        float(with_metrics["weighted_f1"])
        - float(without_metrics["weighted_f1"])
    )

    difference: dict[str, float | int | str] = {
        "experiment": "difference_with_minus_without",
        "training_rows": (
            int(with_metrics["training_rows"])
            - int(without_metrics["training_rows"])
        ),
        "phase5_rows": (
            int(with_metrics["phase5_rows"])
            - int(without_metrics["phase5_rows"])
        ),
        "accuracy": accuracy_difference,
        "macro_f1": macro_f1_difference,
        "weighted_f1": weighted_f1_difference,
    }

    summary_dataframe = pd.DataFrame(
        [
            without_metrics,
            with_metrics,
            difference,
        ]
    )

    summary_dataframe.to_csv(
        OUTPUT_DIR / "phase5_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_class_rows: list[dict[str, float | str]] = []

    for class_name in CLASS_NAMES:
        without_precision = float(
            without_report.loc[
                class_name,
                "precision",
            ]
        )
        with_precision = float(
            with_report.loc[
                class_name,
                "precision",
            ]
        )

        without_recall = float(
            without_report.loc[
                class_name,
                "recall",
            ]
        )
        with_recall = float(
            with_report.loc[
                class_name,
                "recall",
            ]
        )

        without_f1 = float(
            without_report.loc[
                class_name,
                "f1-score",
            ]
        )
        with_f1 = float(
            with_report.loc[
                class_name,
                "f1-score",
            ]
        )

        support = float(
            with_report.loc[
                class_name,
                "support",
            ]
        )

        per_class_rows.append(
            {
                "clause_type": class_name,
                "precision_without_phase5": without_precision,
                "precision_with_phase5": with_precision,
                "precision_change": (
                    with_precision
                    - without_precision
                ),
                "recall_without_phase5": without_recall,
                "recall_with_phase5": with_recall,
                "recall_change": (
                    with_recall
                    - without_recall
                ),
                "f1_without_phase5": without_f1,
                "f1_with_phase5": with_f1,
                "f1_change": (
                    with_f1
                    - without_f1
                ),
                "support": support,
            }
        )

    per_class_dataframe = pd.DataFrame(
        per_class_rows
    )

    per_class_dataframe.to_csv(
        OUTPUT_DIR / "phase5_ablation_per_class.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_lines = [
        "PHASE-5 PROVISIONAL LABEL ABLATION",
        "=" * 80,
        "",
        f"Frozen evaluation clauses: {len(evaluation_data)}",
        (
            "Phase-5 origin label: "
            f"{PHASE5_ORIGIN}"
        ),
        "",
        "SUMMARY",
        "-" * 80,
        summary_dataframe.to_string(index=False),
        "",
        "INTERPRETATION RULE",
        "-" * 80,
        (
            "Positive differences mean that including Phase-5 "
            "provisional labels improved the metric."
        ),
        (
            "Negative differences mean that including Phase-5 "
            "provisional labels reduced the metric."
        ),
        "",
    ]

    (
        OUTPUT_DIR / "phase5_ablation_summary.txt"
    ).write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("PHASE-5 ABLATION COMPLETED")
    print("=" * 70)

    print("\nWithout Phase-5 labels:")
    print(
        f"  Accuracy:    "
        f"{float(without_metrics['accuracy']):.4f}"
    )
    print(
        f"  Macro F1:    "
        f"{float(without_metrics['macro_f1']):.4f}"
    )
    print(
        f"  Weighted F1: "
        f"{float(without_metrics['weighted_f1']):.4f}"
    )

    print("\nWith Phase-5 labels:")
    print(
        f"  Accuracy:    "
        f"{float(with_metrics['accuracy']):.4f}"
    )
    print(
        f"  Macro F1:    "
        f"{float(with_metrics['macro_f1']):.4f}"
    )
    print(
        f"  Weighted F1: "
        f"{float(with_metrics['weighted_f1']):.4f}"
    )

    print("\nDifference, with minus without:")
    print(
        f"  Accuracy:    "
        f"{accuracy_difference:+.4f}"
    )
    print(
        f"  Macro F1:    "
        f"{macro_f1_difference:+.4f}"
    )
    print(
        f"  Weighted F1: "
        f"{weighted_f1_difference:+.4f}"
    )

    print(
        "\nResults saved in:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()