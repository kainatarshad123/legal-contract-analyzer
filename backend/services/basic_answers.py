"""Basic summary, payment, and termination answer builders."""
import re
from typing import Any

from services.contract_analysis import (
    extract_contract_type,
    extract_lease_term,
    extract_missing_fields,
    extract_parties,
    extract_rent_amount,
    extract_start_date,
    find_clauses_by_keywords,
    format_related_clauses,
    get_detected_risk_keywords,
    get_overall_risk,
)


PAYMENT_KEYWORDS = [
    "rent", "monthly rent", "ground rent", "payment", "pay", "paid",
    "payable", "due", "interest", "deposit", "fee", "charges",
    "deductions", "arrears",
]

TERMINATION_KEYWORDS = [
    "termination", "terminate", "earlier determination", "expiration",
    "default", "breach", "arrears", "re-entry", "reentry", "not paid",
    "failure to pay", "failed to comply", "covenants",
    "demise shall absolutely determine", "notice in writing",
]


def build_summary_answer(contract_data: dict[str, Any]) -> str:
    text = contract_data["text"]
    clauses = contract_data["clauses"]

    contract_type = extract_contract_type(text)
    lessor, lessee = extract_parties(text)
    rent_amount = extract_rent_amount(text)
    lease_term = extract_lease_term(text)
    start_date = extract_start_date(text)
    overall_risk = get_overall_risk(clauses)
    risk_keywords = get_detected_risk_keywords(clauses)
    missing_fields = extract_missing_fields(text)

    risk_lines = []

    if "default" in risk_keywords:
        risk_lines.append("Rent/default wording appears in the contract.")

    if "termination" in risk_keywords or "earlier determination" in risk_keywords:
        risk_lines.append("The contract includes termination or early determination wording.")

    if "breach" in risk_keywords:
        risk_lines.append("The contract contains breach-related wording.")

    if not risk_lines:
        risk_lines.append("No major risk signals were detected by the prototype model.")

    missing_text = "\n".join([f"- {field}" for field in missing_fields]) if missing_fields else "- No major missing fields detected."

    risk_text = "\n".join([f"- {item}" for item in risk_lines])

    answer = f"""
Contract Summary

Contract type:
{contract_type}

Parties:
- Lessor: {lessor}
- Lessee: {lessee}

Main purpose:
This document appears to create a lease arrangement where the Lessor grants lease rights over the described premises to the Lessee.

Key terms:
- Lease term: {lease_term}
- Lease start date: {start_date}
- Monthly rent: {rent_amount}

Overall risk level:
{overall_risk}

Main risks detected:
{risk_text}

Missing or incomplete information:
{missing_text}

Important note:
This is a prototype ML-based contract review summary and not legal advice.
"""

    return answer.strip()


def build_payment_answer(contract_data: dict[str, Any]) -> str:
    text = contract_data["text"]
    clauses = contract_data["clauses"]

    payment_clauses = find_clauses_by_keywords(clauses, PAYMENT_KEYWORDS)

    rent_amount = extract_rent_amount(text)

    payment_date = "Not specified in the document."
    interest_rate = "Not specified in the document."

    if re.search(r"interest", text, flags=re.IGNORECASE):
        interest_rate = "Mentioned, but the exact interest rate appears to be blank or unclear."

    risk_level = "Medium" if payment_clauses else "Low"

    answer = f"""
Payment / Rent Review

Main rent obligation:
The Lessee appears to be required to pay monthly ground rent to the Lessor.

Rent amount:
{rent_amount}

Payment date:
{payment_date}

Late payment consequence:
{interest_rate}

Risk level:
{risk_level}

Why this matters:
The payment section appears incomplete because important commercial terms such as the rent amount, due date, and interest rate are not clearly filled in. This can create confusion or future disputes between the Lessor and the Lessee.

Related clauses:
{format_related_clauses(payment_clauses, limit=4)}

Important note:
This is a prototype ML-based contract review response and not legal advice.
"""

    return answer.strip()


def build_termination_answer(contract_data: dict[str, Any]) -> str:
    clauses = contract_data["clauses"]

    termination_clauses = find_clauses_by_keywords(clauses, TERMINATION_KEYWORDS)

    missing_items = [
        "Number of months before rent default applies",
        "Notice requirement before termination",
        "Cure period for the Lessee",
        "Clear procedure for early termination",
    ]

    missing_text = "\n".join([f"- {item}" for item in missing_items])

    answer = f"""
Termination / Default Review

Main issue:
The lease appears to allow early determination or termination if the Lessee fails to pay rent, falls into default, or breaches lease obligations.

Default triggers detected:
- Non-payment of monthly ground rent
- Breach of lease covenants by the Lessee
- Possible early determination of the lease

Risk level:
Medium

Why this may be risky:
The contract contains termination/default wording, but some important details appear blank or unclear. In particular, the default period and process for notice or cure are not clearly stated.

Missing or unclear information:
{missing_text}

Related clauses:
{format_related_clauses(termination_clauses, limit=4)}

Important note:
This is a prototype ML-based contract review response and not legal advice.
"""

    return answer.strip()
