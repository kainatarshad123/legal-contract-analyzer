"""Structured risk and general responses."""

from typing import Any

from services.contract_analysis import (
    get_detected_risk_keywords,
    get_overall_risk,
)
from services.risk_answers import build_risky_clauses_answer
from services.structured_contract_answers import (
    make_related_clause_items,
)


def build_structured_risks(
    contract_data: dict[str, Any],
) -> dict[str, Any]:
    """Build a structured response describing risky clauses."""

    clauses = contract_data.get("clauses", [])

    if not isinstance(clauses, list):
        clauses = []

    risky_clauses = [
        clause
        for clause in clauses
        if isinstance(clause, dict)
        and clause.get("risk_level") in {"Medium", "High"}
    ]

    overall_risk = get_overall_risk(clauses)
    risk_keywords = get_detected_risk_keywords(clauses)

    key_points: list[str] = []

    if risky_clauses:
        key_points.append(
            f"{len(risky_clauses)} medium-risk or high-risk "
            "clause(s) were detected."
        )
    else:
        key_points.append(
            "No medium-risk or high-risk clauses were detected "
            "by the current prototype model."
        )

    if risk_keywords:
        key_points.append(
            "Detected risk signals include: "
            + ", ".join(risk_keywords[:8])
            + "."
        )

    key_points.append(
        "The risk result uses ML prediction together with "
        "keyword and rule validation."
    )

    fallback_answer = build_risky_clauses_answer(
        contract_data
    )

    return {
        "answer_type": "risk",
        "title": "Risky Clauses Review",
        "risk_level": overall_risk,
        "summary": (
            "The model reviewed the contract clauses and identified "
            "clauses that may require closer review."
        ),
        "key_points": key_points,
        "missing_fields": [],
        "related_clauses": make_related_clause_items(
            risky_clauses,
            limit=8,
        ),
        "contract_details": {
            "total_risky_clauses": len(risky_clauses),
            "detected_risk_signals": risk_keywords,
        },
        "answer": fallback_answer,
    }


def build_structured_general(
    contract_data: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    """Build a fallback structured response."""

    clauses = contract_data.get("clauses", [])

    if not isinstance(clauses, list):
        clauses = []

    return {
        "answer_type": "general",
        "title": "Contract Agent Response",
        "risk_level": get_overall_risk(clauses),
        "summary": (
            "The agent can answer questions about contract summaries, "
            "risks, payments, termination, clauses, and missing fields."
        ),
        "key_points": [
            "Try asking: Summarize this contract.",
            "Try asking: What are the risky clauses?",
            "Try asking: What are the payment terms?",
            "Try asking: Explain the termination clause.",
            "Try asking: What information is missing?",
        ],
        "missing_fields": [],
        "related_clauses": [],
        "contract_details": {
            "question": question,
        },
        "answer": (
            "I could not match this question to a supported structured "
            "category. Please ask about the summary, risks, payments, "
            "termination, clauses, or missing information."
        ),
    }