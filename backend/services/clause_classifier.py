"""ML and hybrid clause-type classifier."""

import os
from typing import Any

import joblib

from services.clause_rules import detect_high_precision_clause_rule


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLAUSE_TYPE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml_model",
    "clause_type_model.pkl",
)

CLAUSE_TYPE_VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "ml_model",
    "clause_type_vectorizer.pkl",
)


clause_type_model = None
clause_type_vectorizer = None

try:
    clause_type_model = joblib.load(CLAUSE_TYPE_MODEL_PATH)
    clause_type_vectorizer = joblib.load(CLAUSE_TYPE_VECTORIZER_PATH)
    print("Clause-type model loaded successfully.")
except Exception as error:
    print(
        "Clause-type model not loaded. "
        "Using rule-based clause-type fallback."
    )
    print("Clause-type model error:", error)


HYBRID_ML_CONFIDENCE_THRESHOLD = 0.40
HIGH_CONFIDENCE_THRESHOLD = 0.70
MANUAL_REVIEW_THRESHOLD = 0.55


HYBRID_DISTINCTIVE_RULE_LABELS = {
    "Signature / Execution",
    "Quiet Enjoyment",
    "Assignment / Subletting",
    "Insurance",
    "Liability / Indemnity",
    "Governing Law",
    "Dispute Resolution",
    "Confidentiality",
}


HYBRID_ALLOWED_LABELS = {
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
}


FALLBACK_SOURCES = {
    "hybrid_fallback",
    "hybrid_rule_fallback",
    "fallback",
}


def predict_clause_type_ml(clause_text: str) -> dict[str, Any]:
    """
    Predict a clause category using the trained TF-IDF vectorizer and
    Logistic Regression classifier.
    """

    fallback: dict[str, Any] = {
        "clause_type": "General Contract Clause",
        "clause_type_confidence": 0.0,
        "clause_type_source": "fallback",
    }

    if not isinstance(clause_text, str):
        return fallback

    cleaned_text = clause_text.strip()

    if not cleaned_text:
        return fallback

    if clause_type_model is None or clause_type_vectorizer is None:
        return fallback

    try:
        transformed = clause_type_vectorizer.transform([cleaned_text])
        prediction = clause_type_model.predict(transformed)[0]
        probabilities = clause_type_model.predict_proba(transformed)[0]
        confidence = float(probabilities.max())

        return {
            "clause_type": str(prediction),
            "clause_type_confidence": round(confidence, 4),
            "clause_type_source": "ml",
        }

    except Exception as error:
        print("Clause-type prediction failed:", error)
        return fallback


def build_confidence_review_metadata(
    confidence: float,
    source: str,
) -> dict[str, object]:
    """
    Convert model probability into user-facing confidence and
    manual-review metadata.
    """

    safe_confidence = max(0.0, min(float(confidence), 1.0))

    if safe_confidence >= HIGH_CONFIDENCE_THRESHOLD:
        confidence_label = "High"
    elif safe_confidence >= MANUAL_REVIEW_THRESHOLD:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    needs_manual_review = (
        safe_confidence < MANUAL_REVIEW_THRESHOLD
        or source in FALLBACK_SOURCES
    )

    if source in FALLBACK_SOURCES:
        manual_review_reason = (
            "The classifier used a fallback result, so this clause should "
            "be checked manually."
        )
    elif safe_confidence < MANUAL_REVIEW_THRESHOLD:
        manual_review_reason = (
            "Clause-type confidence is below the manual-review threshold "
            f"of {MANUAL_REVIEW_THRESHOLD:.2f}."
        )
    elif safe_confidence < HIGH_CONFIDENCE_THRESHOLD:
        manual_review_reason = (
            "The prediction has medium confidence. Manual review is optional "
            "but recommended for important clauses."
        )
    else:
        manual_review_reason = (
            "The prediction has high model confidence and does not require "
            "manual review under the current threshold."
        )

    if source == "hybrid_ml":
        confidence_basis = "ML prediction probability"
    elif source == "hybrid_rule":
        confidence_basis = (
            "ML probability associated with a rule-selected clause type"
        )
    elif source == "hybrid_rule_fallback":
        confidence_basis = (
            "ML probability was unavailable or invalid; a rule fallback "
            "selected the clause type"
        )
    else:
        confidence_basis = "Fallback classification confidence"

    review_status = (
        "Needs manual review"
        if needs_manual_review
        else "Manual review not required"
    )

    return {
        "confidence_label": confidence_label,
        "needs_manual_review": needs_manual_review,
        "review_status": review_status,
        "manual_review_reason": manual_review_reason,
        "confidence_basis": confidence_basis,
        "review_threshold": MANUAL_REVIEW_THRESHOLD,
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
    }


def predict_clause_type_hybrid(clause_text: str) -> dict[str, Any]:
    """
    Classify a clause using the ML model and high-precision legal rules.

    Rules may override the ML result when the ML prediction is invalid,
    has low confidence, or the rule belongs to a distinctive category.
    """

    ml_result = predict_clause_type_ml(clause_text)
    rule_result = detect_high_precision_clause_rule(clause_text)

    ml_label = str(
        ml_result.get("clause_type", "")
    ).strip()

    ml_confidence = float(
        ml_result.get("clause_type_confidence", 0.0) or 0.0
    )

    rule_label = str(
        rule_result.get("rule_label", "")
    ).strip()

    rule_matches = rule_result.get("rule_matches", [])

    ml_is_valid = ml_label in HYBRID_ALLOWED_LABELS
    rule_is_valid = rule_label in HYBRID_ALLOWED_LABELS

    use_rule = (
        rule_is_valid
        and (
            not ml_is_valid
            or ml_confidence < HYBRID_ML_CONFIDENCE_THRESHOLD
            or rule_label in HYBRID_DISTINCTIVE_RULE_LABELS
        )
    )

    if use_rule:
        final_label = rule_label
        source = "hybrid_rule"

    elif ml_is_valid:
        final_label = ml_label
        source = "hybrid_ml"

    elif rule_is_valid:
        final_label = rule_label
        source = "hybrid_rule_fallback"

    else:
        final_label = "Other / Unknown"
        source = "hybrid_fallback"

    review_metadata = build_confidence_review_metadata(
        confidence=ml_confidence,
        source=source,
    )

    return {
        "clause_type": final_label,
        "clause_type_confidence": round(ml_confidence, 4),
        "clause_type_source": source,
        "ml_clause_type": ml_label if ml_is_valid else None,
        "rule_clause_type": rule_label if rule_is_valid else None,
        "rule_matches": rule_matches,
        "hybrid_threshold": HYBRID_ML_CONFIDENCE_THRESHOLD,
        **review_metadata,
    }