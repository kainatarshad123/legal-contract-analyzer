"""Select the structured response builder for a user question."""

from typing import Any

from services.structured_contract_answers import (
    build_structured_payment,
    build_structured_summary,
    build_structured_termination,
)
from services.structured_risk_answers import (
    build_structured_general,
    build_structured_risks,
)


def build_structured_missing_fields(
    contract_data: dict[str, Any],
) -> dict[str, Any]:
    """Build a structured response for missing contract information."""

    analysis = contract_data.get("analysis", {})

    if not isinstance(analysis, dict):
        analysis = {}

    missing_fields = analysis.get("missing_fields", [])

    if not isinstance(missing_fields, list):
        missing_fields = []

    cleaned_missing_fields = [
        str(field).strip()
        for field in missing_fields
        if str(field).strip()
    ]

    if cleaned_missing_fields:
        summary = (
            f"The analyzer identified {len(cleaned_missing_fields)} "
            "missing or incomplete contract field(s)."
        )

        key_points = [
            f"Missing or incomplete: {field}"
            for field in cleaned_missing_fields
        ]
    else:
        summary = (
            "The analyzer did not identify any of its currently "
            "supported missing-field categories."
        )

        key_points = [
            (
                "No supported missing fields were detected by the "
                "current extraction rules."
            ),
            (
                "Manual review is still recommended because the "
                "prototype may not detect every omitted term."
            ),
        ]

    return {
        "answer_type": "missing_fields",
        "title": "Missing Information Review",
        "risk_level": analysis.get(
            "overall_risk",
            "Unknown",
        ),
        "summary": summary,
        "key_points": key_points,
        "missing_fields": cleaned_missing_fields,
        "related_clauses": [],
        "contract_details": {
            "total_missing_fields": len(
                cleaned_missing_fields
            ),
        },
        "answer": (
            "Review the identified missing or incomplete fields "
            "before relying on or signing the contract."
            if cleaned_missing_fields
            else (
                "No supported missing fields were detected, but "
                "the contract should still be reviewed manually."
            )
        ),
    }


def build_structured_answer(
    contract_data: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    """Choose the structured response builder for a user question."""

    question_lower = question.lower().strip()

    if any(
        word in question_lower
        for word in [
            "summary",
            "summarize",
            "overview",
        ]
    ):
        return build_structured_summary(contract_data)

    if any(
        word in question_lower
        for word in [
            "payment",
            "rent",
            "fee",
            "deposit",
            "amount",
        ]
    ):
        return build_structured_payment(contract_data)

    if any(
        word in question_lower
        for word in [
            "termination",
            "terminate",
            "default",
            "breach",
            "expiration",
        ]
    ):
        return build_structured_termination(contract_data)

    if any(
        word in question_lower
        for word in [
            "risk",
            "risky",
            "dangerous",
            "problem",
            "clause",
        ]
    ):
        return build_structured_risks(contract_data)

    if any(
        word in question_lower
        for word in [
            "missing",
            "blank",
            "not specified",
            "incomplete",
        ]
    ):
        return build_structured_missing_fields(
            contract_data
        )

    return build_structured_general(
        contract_data,
        question,
    )