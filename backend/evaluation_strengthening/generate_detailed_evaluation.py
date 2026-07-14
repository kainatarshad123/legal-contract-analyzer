from pathlib import Path
import sys
from typing import Any, Sequence

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND_DIR / "ml_model"
OUTPUT_DIR = BACKEND_DIR / "evaluation"

DATA_FILE = MODEL_DIR / "external_test_clauses_final.csv"
MODEL_FILE = MODEL_DIR / "clause_type_model.pkl"
VECTORIZER_FILE = MODEL_DIR / "clause_type_vectorizer.pkl"

sys.path.insert(0, str(BACKEND_DIR))

from validate_hybrid_clause_model import (  # noqa: E402
    ALLOWED_LABELS,
    LOW_CONFIDENCE_THRESHOLD,
    detect_rule_label,
    load_external_data,
)


# ---------------------------------------------------------
# Fixed 22-class order
# ---------------------------------------------------------

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


DISTINCTIVE_RULE_LABELS: set[str] = {
    "Signature / Execution",
    "Quiet Enjoyment",
    "Assignment / Subletting",
    "Insurance",
    "Liability / Indemnity",
    "Governing Law",
    "Dispute Resolution",
    "Confidentiality",
}


# ---------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------

def make_hybrid_predictions(
    texts: Sequence[str],
    model: Any,
    vectorizer: Any,
) -> tuple[list[str], list[str]]:
    """
    Generate final hybrid predictions using the same logic as the
    existing validate_hybrid_clause_model.py script.

    Returns:
        A tuple containing:
        - final hybrid predictions
        - prediction sources, either "ml" or "rule"
    """

    features = vectorizer.transform(texts)
    ml_predictions = model.predict(features)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        ml_confidences = probabilities.max(axis=1)
    else:
        ml_confidences = [0.0] * len(texts)

    hybrid_predictions: list[str] = []
    hybrid_sources: list[str] = []

    for index, text in enumerate(texts):
        rule_label, _ = detect_rule_label(text)

        ml_label = str(ml_predictions[index]).strip()
        confidence = float(ml_confidences[index])

        use_rule = (
            rule_label in ALLOWED_LABELS
            and (
                confidence < LOW_CONFIDENCE_THRESHOLD
                or rule_label in DISTINCTIVE_RULE_LABELS
            )
        )

        if use_rule and rule_label is not None:
            hybrid_predictions.append(rule_label)
            hybrid_sources.append("rule")
        else:
            hybrid_predictions.append(ml_label)
            hybrid_sources.append("ml")

    return hybrid_predictions, hybrid_sources


# ---------------------------------------------------------
# Evaluation output
# ---------------------------------------------------------

def save_classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    model_name: str,
) -> pd.DataFrame:
    """
    Save precision, recall, F1-score, and support for all 22 classes.
    """

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_dataframe = pd.DataFrame(report).transpose()

    report_path = (
        OUTPUT_DIR
        / f"{model_name}_classification_report.csv"
    )

    report_dataframe.to_csv(
        report_path,
        encoding="utf-8-sig",
    )

    return report_dataframe


def save_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    model_name: str,
) -> pd.DataFrame:
    """
    Save the 22 x 22 confusion matrix as CSV and PNG.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
    )

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    matrix_dataframe.index.name = "Actual class"
    matrix_dataframe.columns.name = "Predicted class"

    csv_path = (
        OUTPUT_DIR
        / f"{model_name}_confusion_matrix.csv"
    )

    matrix_dataframe.to_csv(
        csv_path,
        encoding="utf-8-sig",
    )

    figure, axis = plt.subplots(
        figsize=(23, 21)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    display.plot(
        ax=axis,
        xticks_rotation=90,
        values_format="d",
        colorbar=False,
        cmap="Blues",
    )

    axis.set_title(
        f"{model_name.replace('_', ' ').title()} Confusion Matrix",
        fontsize=18,
        pad=20,
    )

    axis.set_xlabel(
        "Predicted Clause Type",
        fontsize=13,
    )

    axis.set_ylabel(
        "Actual Clause Type",
        fontsize=13,
    )

    figure.tight_layout()

    png_path = (
        OUTPUT_DIR
        / f"{model_name}_confusion_matrix.png"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return matrix_dataframe


def calculate_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> dict[str, float]:
    """
    Calculate aggregate evaluation metrics.
    """

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


def evaluate_model(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    model_name: str,
) -> dict[str, float]:
    """
    Generate all detailed evaluation outputs for one prediction method.
    """

    save_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        model_name=model_name,
    )

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        model_name=model_name,
    )

    return calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Frozen evaluation dataset was not found:\n{DATA_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Clause classifier was not found:\n{MODEL_FILE}"
        )

    if not VECTORIZER_FILE.exists():
        raise FileNotFoundError(
            f"Clause vectorizer was not found:\n{VECTORIZER_FILE}"
        )

    print("Loading frozen external evaluation dataset...")

    dataframe = load_external_data()

    print(f"Evaluation clauses loaded: {len(dataframe)}")

    print("Loading clause classifier and vectorizer...")

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    texts = dataframe["clause_text"].tolist()
    actual_labels = dataframe["clause_type"].tolist()

    # Character TF-IDF ML-only predictions
    features = vectorizer.transform(texts)
    character_predictions = model.predict(features)

    # Final hybrid predictions
    hybrid_predictions, hybrid_sources = make_hybrid_predictions(
        texts=texts,
        model=model,
        vectorizer=vectorizer,
    )

    print("\nEvaluating character TF-IDF model...")

    character_metrics = evaluate_model(
        y_true=actual_labels,
        y_pred=character_predictions,
        model_name="character_model",
    )

    print("Evaluating final hybrid classifier...")

    hybrid_metrics = evaluate_model(
        y_true=actual_labels,
        y_pred=hybrid_predictions,
        model_name="final_hybrid",
    )

    predictions_dataframe = dataframe.copy()

    predictions_dataframe["character_prediction"] = (
        character_predictions
    )

    predictions_dataframe["hybrid_prediction"] = (
        hybrid_predictions
    )

    predictions_dataframe["hybrid_source"] = (
        hybrid_sources
    )

    predictions_dataframe.to_csv(
        OUTPUT_DIR
        / "detailed_evaluation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_dataframe = pd.DataFrame(
        [
            {
                "approach": "Character TF-IDF ML only",
                **character_metrics,
                "evaluation_clauses": len(dataframe),
                "ml_predictions": len(dataframe),
                "rule_predictions": 0,
            },
            {
                "approach": "Final hybrid ML + rules",
                **hybrid_metrics,
                "evaluation_clauses": len(dataframe),
                "ml_predictions": hybrid_sources.count("ml"),
                "rule_predictions": hybrid_sources.count("rule"),
            },
        ]
    )

    summary_dataframe.to_csv(
        OUTPUT_DIR
        / "detailed_evaluation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_lines = [
        "DETAILED 22-CLASS EVALUATION",
        "=" * 80,
        "",
        f"Frozen evaluation clauses: {len(dataframe)}",
        f"Class count: {len(CLASS_NAMES)}",
        "",
        "SUMMARY",
        "-" * 80,
        summary_dataframe.to_string(index=False),
        "",
        "FILES GENERATED",
        "-" * 80,
        "character_model_classification_report.csv",
        "character_model_confusion_matrix.csv",
        "character_model_confusion_matrix.png",
        "final_hybrid_classification_report.csv",
        "final_hybrid_confusion_matrix.csv",
        "final_hybrid_confusion_matrix.png",
        "detailed_evaluation_predictions.csv",
        "detailed_evaluation_summary.csv",
        "",
    ]

    summary_path = (
        OUTPUT_DIR
        / "detailed_evaluation_summary.txt"
    )

    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("DETAILED EVALUATION COMPLETED")
    print("=" * 70)

    print("\nCharacter TF-IDF model:")
    print(
        f"  Accuracy:    "
        f"{character_metrics['accuracy']:.4f}"
    )
    print(
        f"  Macro F1:    "
        f"{character_metrics['macro_f1']:.4f}"
    )
    print(
        f"  Weighted F1: "
        f"{character_metrics['weighted_f1']:.4f}"
    )

    print("\nFinal hybrid classifier:")
    print(
        f"  Accuracy:    "
        f"{hybrid_metrics['accuracy']:.4f}"
    )
    print(
        f"  Macro F1:    "
        f"{hybrid_metrics['macro_f1']:.4f}"
    )
    print(
        f"  Weighted F1: "
        f"{hybrid_metrics['weighted_f1']:.4f}"
    )

    print("\nHybrid prediction sources:")
    print(
        f"  ML predictions:   "
        f"{hybrid_sources.count('ml')}"
    )
    print(
        f"  Rule predictions: "
        f"{hybrid_sources.count('rule')}"
    )

    print(
        "\nAll output files were saved in:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()