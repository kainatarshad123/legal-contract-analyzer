"""Contract clause analysis and field-extraction helpers."""

import re
from typing import Any

from services.clause_classifier import predict_clause_type_hybrid
from services.clause_rules import detect_secondary_clause_types
from services.risk_analyzer import (
    detect_party_affected,
    detect_risk_signals,
    final_risk_level,
    generate_recommended_action,
    generate_risk_reason,
    predict_ml_risk,
)
from services.text_processing import (
    clean_display_text,
    has_blank_or_placeholder,
    smart_preview,
    value_or_not_specified,
)


def analyze_clauses(
    raw_clauses: list[str],
) -> list[dict[str, Any]]:
    """Analyze risk, clause type, confidence, and affected party."""

    analyzed_clauses: list[dict[str, Any]] = []

    for index, clause_text in enumerate(raw_clauses, start=1):
        # Risk prediction.
        ml_prediction = predict_ml_risk(clause_text)
        risk_signals = detect_risk_signals(clause_text)

        risk_level = final_risk_level(
            ml_prediction,
            risk_signals,
            clause_text,
        )

        # Hybrid clause-type prediction.
        hybrid_type_result = predict_clause_type_hybrid(clause_text)

        clause_type = str(
            hybrid_type_result.get(
                "clause_type",
                "Other / Unknown",
            )
        )

        clause_type_source = str(
            hybrid_type_result.get(
                "clause_type_source",
                "hybrid_fallback",
            )
        )

        clause_type_confidence = float(
            hybrid_type_result.get(
                "clause_type_confidence",
                0.0,
            )
            or 0.0
        )

        confidence_label = str(
            hybrid_type_result.get(
                "confidence_label",
                "Low",
            )
        )

        needs_manual_review = bool(
            hybrid_type_result.get(
                "needs_manual_review",
                True,
            )
        )

        review_status = str(
            hybrid_type_result.get(
                "review_status",
                (
                    "Needs manual review"
                    if needs_manual_review
                    else "Manual review not required"
                ),
            )
        )

        manual_review_reason = str(
            hybrid_type_result.get(
                "manual_review_reason",
                "The clause classification should be checked manually.",
            )
        )

        confidence_basis = str(
            hybrid_type_result.get(
                "confidence_basis",
                "ML prediction probability",
            )
        )

        review_threshold = float(
            hybrid_type_result.get(
                "review_threshold",
                0.55,
            )
            or 0.55
        )

        high_confidence_threshold = float(
            hybrid_type_result.get(
                "high_confidence_threshold",
                0.70,
            )
            or 0.70
        )

        secondary_clause_types = detect_secondary_clause_types(
            clause_text=clause_text,
            primary_clause_type=clause_type,
        )

        party_affected = detect_party_affected(clause_text)

        risk_reason = generate_risk_reason(
            clause_text=clause_text,
            risk_level=risk_level,
            clause_type=clause_type,
            risk_signals=risk_signals,
        )

        recommended_action = generate_recommended_action(
            clause_type=clause_type,
            risk_level=risk_level,
            clause_text=clause_text,
        )

        analyzed_clauses.append({
            "clause_number": index,
            "text": clean_display_text(clause_text),
            "preview": smart_preview(clause_text),
            "risk_level": risk_level,
            "ml_prediction": ml_prediction,
            "risk_signals": risk_signals,

            # Clause classification.
            "clause_type": clause_type,
            "clause_type_confidence": round(
                clause_type_confidence,
                4,
            ),
            "clause_type_source": clause_type_source,

            # Confidence-based review metadata.
            "confidence_label": confidence_label,
            "needs_manual_review": needs_manual_review,
            "review_status": review_status,
            "manual_review_reason": manual_review_reason,
            "confidence_basis": confidence_basis,
            "review_threshold": review_threshold,
            "high_confidence_threshold": (
                high_confidence_threshold
            ),

            # Hybrid model diagnostics.
            "clause_type_ml_prediction": (
                hybrid_type_result.get("ml_clause_type")
            ),
            "clause_type_rule_prediction": (
                hybrid_type_result.get("rule_clause_type")
            ),
            "clause_type_rule_matches": (
                hybrid_type_result.get("rule_matches", [])
            ),
            "hybrid_threshold": hybrid_type_result.get(
                "hybrid_threshold",
                0.40,
            ),

            # Additional legal analysis.
            "secondary_clause_types": secondary_clause_types,
            "risk_reason": risk_reason,
            "party_affected": party_affected,
            "recommended_action": recommended_action,
        })

    return analyzed_clauses


def extract_contract_type(text: str) -> str:
    """Identify the general contract type using text indicators."""

    lower_text = text.lower()

    if "lease" in lower_text or "rent" in lower_text:
        return "Lease Deed / Rent Agreement"

    if "employment" in lower_text:
        return "Employment Agreement"

    if (
        "non-disclosure" in lower_text
        or "confidentiality" in lower_text
    ):
        return "Non-Disclosure Agreement"

    if "service agreement" in lower_text:
        return "Service Agreement"

    return "Legal Contract"


def extract_parties(text: str) -> tuple[str, str]:
    """Extract lessor and lessee names where available."""

    lessor = "Not specified in the document."
    lessee = "Not specified in the document."

    lessor_match = re.search(
        (
            r"between\s+(.+?)\s+hereinafter\s+called\s+"
            r"['\"]?The Lessor"
        ),
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    lessee_match = re.search(
        (
            r"and\s+(.+?)\s+hereinafter\s+called\s+"
            r"['\"]?The Lessee"
        ),
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if lessor_match:
        possible_lessor = clean_display_text(
            lessor_match.group(1)
        )

        if (
            not has_blank_or_placeholder(possible_lessor)
            and len(possible_lessor) < 120
        ):
            lessor = possible_lessor

    if lessee_match:
        possible_lessee = clean_display_text(
            lessee_match.group(1)
        )

        if (
            not has_blank_or_placeholder(possible_lessee)
            and len(possible_lessee) < 120
        ):
            lessee = possible_lessee

    return lessor, lessee


def extract_rent_amount(text: str) -> str:
    """Extract a monthly rent amount where available."""

    patterns = [
        r"monthly\s+ground\s+rent\s+of\s+Rs\.?\s*([^\s,.]+)",
        r"monthly\s+rent\s+of\s+Rs\.?\s*([^\s,.]+)",
        r"rent\s+of\s+Rs\.?\s*([^\s,.]+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            value = match.group(1)
            return value_or_not_specified(value)

    return "Not specified in the document."


def extract_lease_term(text: str) -> str:
    """Extract the lease duration where available."""

    match = re.search(
        r"for\s+a\s+term\s+of\s+(.+?)\s+years",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        value = match.group(1) + " years"
        return value_or_not_specified(value)

    return "Not specified in the document."


def extract_start_date(text: str) -> str:
    """Extract the lease commencement date where available."""

    match = re.search(
        r"commencing\s+from\s+(.+?)(?:,|\s+but|\s+and)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        value = match.group(1)
        return value_or_not_specified(value)

    return "Not specified in the document."


def extract_missing_fields(text: str) -> list[str]:
    """Return contract fields that appear missing or incomplete."""

    missing: list[str] = []

    lessor, lessee = extract_parties(text)

    if "Not specified" in lessor:
        missing.append("Lessor name")

    if "Not specified" in lessee:
        missing.append("Lessee name")

    if "Not specified" in extract_rent_amount(text):
        missing.append("Monthly rent amount")

    if "Not specified" in extract_lease_term(text):
        missing.append("Lease duration")

    if "Not specified" in extract_start_date(text):
        missing.append("Lease start date")

    if has_blank_or_placeholder(text):
        missing.append(
            "Some contract blanks/placeholders are still unfilled"
        )

    return missing


def get_overall_risk(
    analyzed_clauses: list[dict[str, Any]],
) -> str:
    """Calculate the highest risk level found in the contract."""

    levels = [
        str(clause.get("risk_level", "Low"))
        for clause in analyzed_clauses
    ]

    if "High" in levels:
        return "High"

    if "Medium" in levels:
        return "Medium"

    return "Low"


def find_clauses_by_keywords(
    analyzed_clauses: list[dict[str, Any]],
    keywords: list[str],
) -> list[dict[str, Any]]:
    """Find analyzed clauses containing any supplied keyword."""

    results: list[dict[str, Any]] = []

    normalized_keywords = [
        keyword.lower()
        for keyword in keywords
        if keyword.strip()
    ]

    for clause in analyzed_clauses:
        lower_text = str(
            clause.get("text", "")
        ).lower()

        if any(
            keyword in lower_text
            for keyword in normalized_keywords
        ):
            results.append(clause)

    return results


def get_detected_risk_keywords(
    analyzed_clauses: list[dict[str, Any]],
) -> list[str]:
    """Return a unique sorted list of detected risk keywords."""

    keywords: list[str] = []

    for clause in analyzed_clauses:
        risk_signals = clause.get("risk_signals", [])

        if not isinstance(risk_signals, list):
            continue

        for signal in risk_signals:
            if not isinstance(signal, dict):
                continue

            keyword = signal.get("keyword")

            if keyword:
                keywords.append(str(keyword))

    return sorted(set(keywords))


def format_related_clauses(
    clauses: list[dict[str, Any]],
    limit: int = 5,
) -> str:
    """Format related clauses for contract Q&A responses."""

    if not clauses:
        return "No directly related clauses were found."

    lines: list[str] = []

    for clause in clauses[:limit]:
        clause_number = clause.get("clause_number", "Unknown")
        risk_level = clause.get("risk_level", "Unknown")
        preview = clause.get("preview", "")

        lines.append(
            f"- Clause {clause_number} — "
            f"{risk_level} Risk: {preview}"
        )

    return "\n".join(lines)