from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ml_model"

DATA_FILE = MODEL_DIR / "external_test_clauses_final.csv"
MODEL_FILE = MODEL_DIR / "clause_type_model.pkl"
VECTORIZER_FILE = MODEL_DIR / "clause_type_vectorizer.pkl"

PREDICTIONS_FILE = MODEL_DIR / "external_test_predictions.csv"
ERRORS_FILE = MODEL_DIR / "external_test_errors.csv"
REPORT_FILE = MODEL_DIR / "external_test_report.txt"


def main() -> None:
    # Verify required files.
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found:\n{DATA_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    if not VECTORIZER_FILE.exists():
        raise FileNotFoundError(
            f"Vectorizer not found:\n{VECTORIZER_FILE}"
        )

    # Load evaluation data.
    dataframe = pd.read_csv(DATA_FILE)

    required_columns = {
        "clause_text",
        "clause_type",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Clean clause text.
    dataframe["clause_text"] = (
        dataframe["clause_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Clean the original clause label.
    dataframe["clause_type"] = (
        dataframe["clause_type"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Use corrected labels when available.
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

    # Keep only rows marked for evaluation.
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

    # Remove empty clauses and empty labels.
    dataframe = dataframe[
        (dataframe["clause_text"] != "")
        & (dataframe["clause_type"] != "")
    ].copy()

    if dataframe.empty:
        raise ValueError(
            "No usable labeled rows were found. "
            "Check corrected_clause_type and include_in_evaluation."
        )

    # Load model and vectorizer.
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    # Convert clause text into TF-IDF features.
    features = vectorizer.transform(
        dataframe["clause_text"]
    )

    # Predict clause types.
    predictions = model.predict(features)

    dataframe["predicted_clause_type"] = predictions

    dataframe["is_correct"] = (
        dataframe["clause_type"]
        == dataframe["predicted_clause_type"]
    )

    # Calculate model confidence.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        dataframe["ml_confidence"] = probabilities.max(axis=1)
    else:
        dataframe["ml_confidence"] = pd.NA

    # Overall metrics.
    accuracy = accuracy_score(
        dataframe["clause_type"],
        dataframe["predicted_clause_type"],
    )

    macro_f1 = f1_score(
        dataframe["clause_type"],
        dataframe["predicted_clause_type"],
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        dataframe["clause_type"],
        dataframe["predicted_clause_type"],
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        dataframe["clause_type"],
        dataframe["predicted_clause_type"],
        zero_division=0,
    )

    if dataframe["ml_confidence"].notna().any():
        average_confidence = float(
            dataframe["ml_confidence"].mean()
        )
    else:
        average_confidence = None

    # Save all predictions.
    dataframe.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # Save incorrect predictions only.
    errors = dataframe[
        ~dataframe["is_correct"]
    ].copy()

    errors.to_csv(
        ERRORS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # Build overall report.
    summary_lines = [
        "EXTERNAL CLAUSE-TYPE EVALUATION",
        "=" * 50,
        f"Total labeled clauses: {len(dataframe)}",
        f"Correct predictions: {int(dataframe['is_correct'].sum())}",
        f"Incorrect predictions: {len(errors)}",
        f"Accuracy: {accuracy:.4f}",
        f"Macro F1: {macro_f1:.4f}",
        f"Weighted F1: {weighted_f1:.4f}",
    ]

    if average_confidence is not None:
        summary_lines.append(
            f"Average ML confidence: {average_confidence:.4f}"
        )

    summary_lines.extend(
        [
            "",
            "CLASSIFICATION REPORT",
            "=" * 50,
            report,
        ]
    )

    # Performance by source PDF.
    source_report_lines = [
        "",
        "PERFORMANCE BY SOURCE",
        "=" * 50,
    ]

    if "source" in dataframe.columns:
        for source_name, group in dataframe.groupby("source"):
            source_accuracy = accuracy_score(
                group["clause_type"],
                group["predicted_clause_type"],
            )

            source_macro_f1 = f1_score(
                group["clause_type"],
                group["predicted_clause_type"],
                average="macro",
                zero_division=0,
            )

            if group["ml_confidence"].notna().any():
                source_confidence = float(
                    group["ml_confidence"].mean()
                )
            else:
                source_confidence = None

            source_report_lines.extend(
                [
                    "",
                    str(source_name),
                    f"Clauses: {len(group)}",
                    f"Correct: {int(group['is_correct'].sum())}",
                    f"Incorrect: {int((~group['is_correct']).sum())}",
                    f"Accuracy: {source_accuracy:.4f}",
                    f"Macro F1: {source_macro_f1:.4f}",
                ]
            )

            if source_confidence is not None:
                source_report_lines.append(
                    f"Average confidence: {source_confidence:.4f}"
                )
    else:
        source_report_lines.append(
            "No source column was found in the CSV."
        )

    summary_lines.extend(source_report_lines)

    report_text = "\n".join(summary_lines)

    # Save the report.
    REPORT_FILE.write_text(
        report_text,
        encoding="utf-8",
    )

    # Display results.
    print(report_text)

    print("\nFiles created:")
    print(PREDICTIONS_FILE)
    print(ERRORS_FILE)
    print(REPORT_FILE)


if __name__ == "__main__":
    main()