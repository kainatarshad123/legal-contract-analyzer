from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ml_model"

DATASET_FILE = (
    MODEL_DIR
    / "combined_clause_type_dataset.csv"
)
MODEL_FILE = MODEL_DIR / "clause_type_model.pkl"
VECTORIZER_FILE = MODEL_DIR / "clause_type_vectorizer.pkl"
REPORT_FILE = MODEL_DIR / "clause_type_training_report.txt"


def load_dataset() -> pd.DataFrame:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}\n"
            "Run prepare_clause_type_dataset.py first."
        )

    dataframe = pd.read_csv(DATASET_FILE)

    required_columns = {
        "clause_text",
        "clause_type",
        "source_split",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    dataframe = dataframe.dropna(
        subset=["clause_text", "clause_type", "source_split"]
    ).copy()

    dataframe["clause_text"] = (
        dataframe["clause_text"]
        .astype(str)
        .str.strip()
    )

    dataframe["clause_type"] = (
        dataframe["clause_type"]
        .astype(str)
        .str.strip()
    )

    dataframe["source_split"] = (
        dataframe["source_split"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    dataframe = dataframe[
        dataframe["clause_text"].str.len() > 0
    ].reset_index(drop=True)

    return dataframe


def split_dataset(dataframe: pd.DataFrame):
    """
    Preserve the original LEDGAR train, validation and test splits.
    """

    train_data = dataframe[
        dataframe["source_split"] == "train"
    ].copy()

    validation_data = dataframe[
        dataframe["source_split"] == "validation"
    ].copy()

    test_data = dataframe[
        dataframe["source_split"] == "test"
    ].copy()

    if train_data.empty:
        raise ValueError("Training split is empty.")

    if validation_data.empty:
        raise ValueError("Validation split is empty.")

    if test_data.empty:
        raise ValueError("Test split is empty.")

    return train_data, validation_data, test_data


def evaluate_model(
    model: Pipeline,
    dataset_name: str,
    data: pd.DataFrame,
) -> str:
    texts = data["clause_text"]
    actual_labels = data["clause_type"]

    predicted_labels = model.predict(texts)

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels,
    )

    report = classification_report(
        actual_labels,
        predicted_labels,
        digits=4,
        zero_division=0,
    )

    labels = sorted(actual_labels.unique())

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=labels,
    )

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[f"actual:{label}" for label in labels],
        columns=[f"predicted:{label}" for label in labels],
    )

    output = [
        "=" * 80,
        f"{dataset_name.upper()} RESULTS",
        "=" * 80,
        f"Examples: {len(data)}",
        f"Accuracy: {accuracy:.4f}",
        "",
        "Classification report:",
        report,
        "",
        "Confusion matrix:",
        matrix_dataframe.to_string(),
        "",
    ]

    return "\n".join(output)


def show_low_confidence_examples(
    model: Pipeline,
    test_data: pd.DataFrame,
    limit: int = 20,
) -> str:
    probabilities = model.predict_proba(
        test_data["clause_text"]
    )

    predictions = model.predict(
        test_data["clause_text"]
    )

    confidence_scores = probabilities.max(axis=1)

    result = test_data[
        ["clause_text", "clause_type"]
    ].copy()

    result["predicted_type"] = predictions
    result["confidence"] = confidence_scores

    result = result.sort_values(
        by="confidence",
        ascending=True,
    ).head(limit)

    output = [
        "=" * 80,
        "LOW-CONFIDENCE TEST EXAMPLES",
        "=" * 80,
    ]

    for index, row in result.iterrows():
        output.append(
            f"\nActual: {row['clause_type']}"
        )
        output.append(
            f"Predicted: {row['predicted_type']}"
        )
        output.append(
            f"Confidence: {row['confidence']:.4f}"
        )
        output.append(
            f"Text: {row['clause_text'][:500]}"
        )
        output.append("-" * 80)

    return "\n".join(output)


def main() -> None:
    print("Loading clause-type dataset...")

    dataframe = load_dataset()

    print(f"Total examples: {len(dataframe)}")
    print(f"Clause categories: {dataframe['clause_type'].nunique()}")

    print("\nExamples per category:")
    print(
        dataframe["clause_type"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    train_data, validation_data, test_data = split_dataset(
        dataframe
    )

    print("\nDataset splits:")
    print(f"Train: {len(train_data)}")
    print(f"Validation: {len(validation_data)}")
    print(f"Test: {len(test_data)}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=30000,
        sublinear_tf=True,
    )

    classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
    solver="lbfgs",
)

    model = Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ]
    )

    print("\nTraining clause-type model...")

    model.fit(
        train_data["clause_text"],
        train_data["clause_type"],
    )

    print("Training completed.")

    validation_report = evaluate_model(
        model=model,
        dataset_name="validation",
        data=validation_data,
    )

    test_report = evaluate_model(
        model=model,
        dataset_name="test",
        data=test_data,
    )

    low_confidence_report = show_low_confidence_examples(
        model=model,
        test_data=test_data,
        limit=20,
    )

    complete_report = "\n".join(
        [
            validation_report,
            test_report,
            low_confidence_report,
        ]
    )

    print("\n" + validation_report)
    print("\n" + test_report)

    REPORT_FILE.write_text(
        complete_report,
        encoding="utf-8",
    )

    trained_vectorizer = model.named_steps["vectorizer"]
    trained_classifier = model.named_steps["classifier"]

    joblib.dump(
        trained_classifier,
        MODEL_FILE,
    )

    joblib.dump(
        trained_vectorizer,
        VECTORIZER_FILE,
    )

    print("\nFiles saved successfully:")
    print(f"Model: {MODEL_FILE}")
    print(f"Vectorizer: {VECTORIZER_FILE}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()