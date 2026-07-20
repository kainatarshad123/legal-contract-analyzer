"""Risk, missing-field, and general contract answer builders."""

from typing import Any

from services.basic_answers import (
    build_payment_answer,
    build_summary_answer,
    build_termination_answer,
)
from services.contract_analysis import (
    extract_missing_fields,
    format_related_clauses,
)


def build_risky_clauses_answer(contract_data: dict[str, Any]) -> str:
    clauses = contract_data["clauses"]

    risky_clauses = [
        clause for clause in clauses
        if clause["risk_level"] in ["Medium", "High"]
    ]

    if not risky_clauses:
        return """
Risky Clauses Review

No medium-risk or high-risk clauses were detected by the current prototype model.

Important note:
This is a prototype ML-based contract review response and not legal advice.
""".strip()

    lines = []

    for clause in risky_clauses[:8]:
        signals = clause.get("risk_signals", [])

        if signals:
            reasons = ", ".join([signal["keyword"] for signal in signals[:3]])
        else:
            reasons = "General risk wording detected."

        explanation = explain_clause_risk(clause)

        lines.append(
            f"""
Clause {clause['clause_number']} — {clause['risk_level']} Risk

Reason:
{reasons}

Explanation:
{explanation}

ML prediction:
{clause['ml_prediction']}

Preview:
{clause['preview']}
""".strip()
        )

    answer = f"""
Risky Clauses Review

I found {len(risky_clauses)} medium-risk or high-risk clause(s) in this contract.

{chr(10).join(lines)}

Important note:
This is a prototype ML-based contract review response and not legal advice.
"""

    return answer.strip()


def explain_clause_risk(clause: dict[str, Any]) -> str:
    text = clause["text"].lower()

    if "rent" in text and ("default" in text or "arrears" in text or "not paid" in text):
        return "This clause may create risk because it connects unpaid rent with default consequences."

    if "termination" in text or "earlier determination" in text:
        return "This clause may create risk because it allows the lease to end earlier than the full term."

    if "breach" in text or "covenants" in text:
        return "This clause may create risk because breach of obligations can trigger legal or contractual consequences."

    if "interest" in text:
        return "This clause may create risk because it mentions interest or late payment consequences, but the exact amount may be unclear."

    if "liability" in text or "indemnify" in text or "indemnification" in text:
        return "This clause may create risk because it may shift liability or financial responsibility to one party."

    return "This clause contains wording that the prototype model identified as requiring review."


def build_general_answer(
    contract_data: dict[str, Any],
    question: str,
) -> str:
    summary = build_summary_answer(contract_data)

    answer = f"""
I can help with contract-specific questions such as:

- Summarize this contract
- What are the risky clauses?
- What are the payment terms?
- Explain the termination clause
- What information is missing?

For now, here is the main contract summary:

{summary}
"""

    return answer.strip()


def build_missing_fields_answer(contract_data: dict[str, Any]) -> str:
    text = contract_data["text"]
    missing_fields = extract_missing_fields(text)

    if not missing_fields:
        return """
Missing Information Review

No major missing fields were detected by the current prototype model.

Important note:
This is a prototype ML-based contract review response and not legal advice.
""".strip()

    missing_text = "\n".join([f"- {field}" for field in missing_fields])

    return f"""
Missing Information Review

The following information appears missing, blank, or incomplete:

{missing_text}

Why this matters:
Blank fields in legal contracts can create confusion, disputes, or enforcement issues because the parties may not have clearly agreed on those terms.

Important note:
This is a prototype ML-based contract review response and not legal advice.
""".strip()


def answer_contract_question(
    contract_data: dict[str, Any],
    question: str,
) -> str:
    question_lower = question.lower()

    if any(word in question_lower for word in ["summary", "summarize", "overview"]):
        return build_summary_answer(contract_data)

    if any(word in question_lower for word in ["payment", "rent", "fee", "deposit", "amount"]):
        return build_payment_answer(contract_data)

    if any(word in question_lower for word in ["termination", "terminate", "default", "breach", "expiration"]):
        return build_termination_answer(contract_data)

    if any(word in question_lower for word in ["risk", "risky", "dangerous", "problem", "clause"]):
        return build_risky_clauses_answer(contract_data)

    if any(word in question_lower for word in ["missing", "blank", "not specified", "incomplete"]):
        return build_missing_fields_answer(contract_data)

    return build_general_answer(contract_data, question)
