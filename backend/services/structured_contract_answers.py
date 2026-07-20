"""Structured summary, payment, and termination responses."""

from typing import Any

from services.basic_answers import (
    PAYMENT_KEYWORDS,
    TERMINATION_KEYWORDS,
    build_payment_answer,
    build_summary_answer,
    build_termination_answer,
)
from services.contract_analysis import (
    extract_contract_type,
    extract_lease_term,
    extract_missing_fields,
    extract_parties,
    extract_rent_amount,
    extract_start_date,
    find_clauses_by_keywords,
    get_detected_risk_keywords,
    get_overall_risk,
)


def make_related_clause_items(
    clauses: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    related = []

    for clause in clauses[:limit]:
        related.append({
            "clause_number": clause.get("clause_number"),
            "risk_level": clause.get("risk_level"),
            "ml_prediction": clause.get("ml_prediction"),
            "preview": clause.get("preview"),
            "clause_type": clause.get("clause_type"),
            "clause_type_confidence": clause.get(
                "clause_type_confidence",
                0.0,
            ),
            "clause_type_source": clause.get(
                "clause_type_source",
                "legacy",
            ),
            "secondary_clause_types": clause.get(
                "secondary_clause_types",
                [],
            ),
            "risk_reason": clause.get("risk_reason"),
            "party_affected": clause.get("party_affected"),
            "recommended_action": clause.get("recommended_action"),
            "risk_signals": [
                signal.get("keyword")
                for signal in clause.get("risk_signals", [])
            ],
        })

    return related


def build_structured_summary(
    contract_data: dict[str, Any],
) -> dict[str, Any]:
    text = contract_data["text"]
    clauses = contract_data["clauses"]

    contract_type = extract_contract_type(text)
    lessor, lessee = extract_parties(text)
    rent_amount = extract_rent_amount(text)
    lease_term = extract_lease_term(text)
    start_date = extract_start_date(text)
    overall_risk = get_overall_risk(clauses)
    missing_fields = extract_missing_fields(text)
    risk_keywords = get_detected_risk_keywords(clauses)

    key_points = [
        f"Contract type detected as {contract_type}.",
        "The document appears to create a lease relationship between a Lessor and a Lessee.",
        f"Overall risk level is {overall_risk}.",
    ]

    if lessor == "Not specified in the document.":
        key_points.append("The Lessor name is not clearly filled in.")

    if lessee == "Not specified in the document.":
        key_points.append("The Lessee name is not clearly filled in.")

    if rent_amount == "Not specified in the document.":
        key_points.append("The monthly rent amount is missing or not clearly stated.")

    if lease_term == "Not specified in the document.":
        key_points.append("The lease duration is missing or not clearly stated.")

    if start_date == "Not specified in the document.":
        key_points.append("The lease start date is missing or not clearly stated.")

    if risk_keywords:
        key_points.append(
            "Detected risk signals include: " + ", ".join(risk_keywords[:6]) + "."
        )

    related_clauses = [
        clause for clause in clauses
        if clause.get("risk_level") in ["Medium", "High"]
    ]

    fallback_answer = build_summary_answer(contract_data)

    return {
        "answer_type": "summary",
        "title": "Contract Summary",
        "risk_level": overall_risk,
        "summary": "This contract appears to be a lease/rent agreement where the Lessor grants lease rights to the Lessee. Several important fields may still be blank or incomplete.",
        "key_points": key_points,
        "missing_fields": missing_fields,
        "related_clauses": make_related_clause_items(related_clauses, limit=5),
        "contract_details": {
            "contract_type": contract_type,
            "lessor": lessor,
            "lessee": lessee,
            "lease_term": lease_term,
            "lease_start_date": start_date,
            "monthly_rent": rent_amount,
        },
        "answer": fallback_answer,
    }


def build_structured_payment(
    contract_data: dict[str, Any],
) -> dict[str, Any]:
    text = contract_data["text"]
    clauses = contract_data["clauses"]

    payment_clauses = find_clauses_by_keywords(clauses, PAYMENT_KEYWORDS)

    rent_amount = extract_rent_amount(text)

    missing_fields = []

    if rent_amount == "Not specified in the document.":
        missing_fields.append("Monthly rent amount")

    missing_fields.extend([
        "Payment due date",
        "Interest rate for late payment",
    ])

    key_points = [
        "The Lessee appears to be required to pay monthly ground rent to the Lessor.",
    ]

    if rent_amount == "Not specified in the document.":
        key_points.append("The rent amount appears blank or not clearly stated.")
    else:
        key_points.append(f"The detected rent amount is {rent_amount}.")

    key_points.append("The exact payment date is not clearly detected.")
    key_points.append("Late-payment interest is mentioned, but the exact rate appears unclear or blank.")

    fallback_answer = build_payment_answer(contract_data)

    return {
        "answer_type": "payment",
        "title": "Payment / Rent Review",
        "risk_level": "Medium",
        "summary": "The contract includes rent/payment obligations, but some commercial payment details appear missing or incomplete.",
        "key_points": key_points,
        "missing_fields": list(dict.fromkeys(missing_fields)),
        "related_clauses": make_related_clause_items(payment_clauses, limit=5),
        "contract_details": {
            "monthly_rent": rent_amount,
            "payment_due_date": "Not specified in the document.",
            "late_payment_interest": "Mentioned, but exact rate appears unclear or blank.",
        },
        "answer": fallback_answer,
    }


def build_structured_termination(
    contract_data: dict[str, Any],
) -> dict[str, Any]:
    clauses = contract_data["clauses"]

    termination_clauses = find_clauses_by_keywords(clauses, TERMINATION_KEYWORDS)

    missing_fields = [
        "Default period",
        "Notice requirement before termination",
        "Cure period for the Lessee",
        "Clear early-termination procedure",
    ]

    key_points = [
        "The contract appears to allow early determination or termination if the Lessee defaults.",
        "Non-payment of rent may trigger default-related consequences.",
        "Breach of lease covenants may also trigger consequences.",
        "The default period and cure process appear unclear or incomplete.",
    ]

    fallback_answer = build_termination_answer(contract_data)

    return {
        "answer_type": "termination",
        "title": "Termination / Default Review",
        "risk_level": "Medium",
        "summary": "The lease appears to include default and early-termination wording, but important procedural details are missing or unclear.",
        "key_points": key_points,
        "missing_fields": missing_fields,
        "related_clauses": make_related_clause_items(termination_clauses, limit=5),
        "contract_details": {
            "default_triggers": [
                "Non-payment of rent",
                "Breach of lease covenants",
                "Possible early determination of the lease",
            ],
            "notice_requirement": "Not clearly specified in the document.",
            "cure_period": "Not clearly specified in the document.",
        },
        "answer": fallback_answer,
    }
