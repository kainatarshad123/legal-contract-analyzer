"""Gemini-backed contract answer builder."""

from typing import Any

from services.contract_analysis import get_overall_risk
from services.gemini_client import (
    LEGAL_ASSISTANT_SYSTEM_PROMPT,
    call_gemini,
)
from services.structured_risk_answers import build_structured_general


def build_llm_general_answer(
    contract_data: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    contract_text = (contract_data.get("text") or "").strip()
    truncated_text = contract_text[:12000]

    prompt = f"""
{LEGAL_ASSISTANT_SYSTEM_PROMPT}

Contract text may be truncated for length:
---
{truncated_text}
---

User question:
{question}

Answer:
"""

    try:
        answer_text = call_gemini(prompt)

    except Exception as error:
        print("Gemini general-answer error:", error)
        fallback = build_structured_general(contract_data, question)
        fallback["summary"] = (
            "Gemini is temporarily unavailable or the API key is not configured correctly. "
            "I am showing the standard contract response instead. Check GEMINI_API_KEY, "
            "make sure the key starts with AIza, and restart the backend."
        )
        fallback["answer"] = fallback["summary"]
        return fallback

    return {
        "answer_type": "general_legal",
        "title": "Legal Q&A",
        "risk_level": get_overall_risk(contract_data["clauses"]),
        "summary": answer_text,
        "key_points": [],
        "missing_fields": [],
        "related_clauses": [],
        "contract_details": {},
        "answer": answer_text,
        "disclaimer": (
            "This is general information, not legal advice. Consult a licensed "
            "attorney for advice about your specific situation."
        ),
    }
