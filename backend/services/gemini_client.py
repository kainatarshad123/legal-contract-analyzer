"""Gemini API client for legal-contract Q&A and clause explanations."""

import json
import os
from typing import Any

from services.rag_service import format_rag_context


GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash-lite",
)


LEGAL_ASSISTANT_SYSTEM_PROMPT = """
You are a legal-contract assistant embedded in a contract review tool.

Rules you must follow:
- Answer using the provided contract text whenever the question concerns
  the specific contract.
- You may answer general legal knowledge questions, such as what a legal
  term means or how a clause usually operates.
- If the question is unrelated to law, contracts, leases, clauses,
  agreements, or legal risk, explain that you only handle legal and
  contract-related questions.
- Do not invent contract terms.
- If the contract is silent on something, say so plainly.
- Use clear and concise plain language.
- Always state that the response is general information and not legal advice.
""".strip()


CLAUSE_EXPLANATION_SYSTEM_PROMPT = """
You are a legal-contract assistant explaining one contract clause.

Explain the clause in plain English for a non-lawyer.

Rules:
- Use only the clause and metadata provided.
- Do not invent facts, obligations, rights, dates, amounts, or parties.
- Distinguish what the clause explicitly states from general observations.
- Explain the practical effect of the clause.
- Identify meaningful risks without exaggerating them.
- Suggest useful questions a person may raise with a qualified lawyer.
- Do not give definitive legal advice.
- Return only valid JSON.
""".strip()


RAG_SYSTEM_PROMPT = """
You are a legal-contract assistant answering a question about one saved contract.

You will receive only a small set of contract excerpts selected by a retrieval system.

Mandatory rules:
- Use only the retrieved contract excerpts.
- Do not rely on unstated terms or outside assumptions.
- Do not invent names, dates, amounts, obligations, rights, remedies, or notice periods.
- When the retrieved excerpts are insufficient, say exactly that the retrieved
  contract sections do not contain enough information to answer the question.
- Clearly distinguish explicit contract wording from cautious interpretation.
- Use plain English.
- Do not give definitive legal advice.
- Return only valid JSON.
""".strip()


def _get_api_key() -> str:
    """Return the configured Gemini API key."""

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not api_key:
        raise ValueError(
            "Gemini API key is missing. In CMD, run: "
            "set GEMINI_API_KEY=YOUR_REAL_KEY, then restart uvicorn."
        )

    return api_key


def _generate_content(prompt: str) -> str:
    """Generate text using the installed Gemini SDK."""

    api_key = _get_api_key()

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        answer_text = (
            getattr(response, "text", None)
            or ""
        ).strip()

        if answer_text:
            return answer_text

        raise ValueError("Empty response from Gemini.")

    except ImportError:
        pass

    try:
        import google.generativeai as generativeai

        generativeai.configure(api_key=api_key)

        model = generativeai.GenerativeModel(
            GEMINI_MODEL
        )

        response = model.generate_content(prompt)

        answer_text = (
            getattr(response, "text", None)
            or ""
        ).strip()

        if answer_text:
            return answer_text

        raise ValueError("Empty response from Gemini.")

    except ImportError as import_error:
        raise ImportError(
            "Gemini SDK is not installed. Run: "
            "pip install google-genai"
        ) from import_error


def call_gemini(prompt: str) -> str:
    """Call Gemini for general legal-contract Q&A."""

    complete_prompt = (
        f"{LEGAL_ASSISTANT_SYSTEM_PROMPT}\n\n"
        f"USER REQUEST:\n{prompt}"
    )

    return _generate_content(complete_prompt)


def _remove_json_code_fence(text: str) -> str:
    """Remove Markdown JSON code fences from Gemini output."""

    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def _normalize_string_list(value: Any) -> list[str]:
    """Convert an unknown value into a clean list of strings."""

    if not isinstance(value, list):
        return []

    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]


def answer_contract_question_with_rag(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    filename: str | None,
) -> dict[str, Any]:
    """Answer a contract question using only retrieved chunks."""

    if not retrieved_chunks:
        fallback_answer = (
            "The retrieved contract sections do not contain enough "
            "information to answer that question."
        )

        return {
            "summary": fallback_answer,
            "answer": fallback_answer,
            "key_points": [],
            "related_clauses": [],
        }

    context = format_rag_context(retrieved_chunks)

    prompt = f"""
{RAG_SYSTEM_PROMPT}

CONTRACT FILE:
{filename or "Saved contract"}

USER QUESTION:
{question}

RETRIEVED CONTRACT CONTEXT:
{context}

Return exactly this JSON structure:

{{
  "summary": "One concise sentence answering the question.",
  "answer": "A clear grounded answer based only on the retrieved excerpts.",
  "key_points": [
    "Important point supported by the retrieved excerpts"
  ],
  "related_clauses": [
    {{
      "clause_number": 1,
      "clause_type": "Clause type",
      "reason": "Why this clause supports the answer"
    }}
  ]
}}

If the context is insufficient, both "summary" and "answer" must say:
"The retrieved contract sections do not contain enough information to answer that question."

Do not add Markdown or any text outside the JSON object.
""".strip()

    raw_response = _generate_content(prompt)
    cleaned_response = _remove_json_code_fence(
        raw_response
    )

    try:
        parsed = json.loads(cleaned_response)
    except json.JSONDecodeError:
        return {
            "summary": raw_response,
            "answer": raw_response,
            "key_points": [],
            "related_clauses": [],
        }

    related_clauses = parsed.get("related_clauses")

    if not isinstance(related_clauses, list):
        related_clauses = []

    normalized_related: list[dict[str, Any]] = []

    for item in related_clauses:
        if not isinstance(item, dict):
            continue

        normalized_related.append(
            {
                "clause_number": item.get("clause_number"),
                "clause_type": str(
                    item.get("clause_type")
                    or "Not classified"
                ).strip(),
                "reason": str(
                    item.get("reason")
                    or ""
                ).strip(),
            }
        )

    answer = str(
        parsed.get("answer")
        or parsed.get("summary")
        or "No answer was returned."
    ).strip()

    summary = str(
        parsed.get("summary")
        or answer
    ).strip()

    return {
        "summary": summary,
        "answer": answer,
        "key_points": _normalize_string_list(
            parsed.get("key_points")
        ),
        "related_clauses": normalized_related,
    }


def explain_clause_with_gemini(
    clause: dict[str, Any],
) -> dict[str, Any]:
    """Explain one analyzed contract clause in plain English."""

    clause_number = clause.get(
        "clause_number",
        "Unknown",
    )

    clause_text = str(
        clause.get("text")
        or clause.get("preview")
        or ""
    ).strip()

    clause_type = str(
        clause.get("clause_type")
        or "Not classified"
    )

    risk_level = str(
        clause.get("risk_level")
        or "Unknown"
    )

    party_affected = str(
        clause.get("party_affected")
        or "Not identified"
    )

    confidence_label = str(
        clause.get("confidence_label")
        or "Unknown"
    )

    needs_manual_review = bool(
        clause.get("needs_manual_review", False)
    )

    prompt = f"""
{CLAUSE_EXPLANATION_SYSTEM_PROMPT}

CLAUSE METADATA:
- Clause number: {clause_number}
- Clause type: {clause_type}
- Risk level: {risk_level}
- Party affected: {party_affected}
- Classification confidence: {confidence_label}
- Needs manual review: {needs_manual_review}

CLAUSE TEXT:
{clause_text}

Return exactly this JSON structure:

{{
  "plain_english_explanation": "Clear explanation for a non-lawyer.",
  "practical_effect": "What this clause may mean in practice.",
  "main_risks": [
    "Risk or concern supported by the clause"
  ],
  "questions_to_consider": [
    "Question the user may raise with a qualified lawyer"
  ],
  "important_note": "Anything important that is unclear or absent."
}}

Do not add Markdown, commentary, or text outside the JSON object.
""".strip()

    raw_response = _generate_content(prompt)
    cleaned_response = _remove_json_code_fence(
        raw_response
    )

    try:
        parsed = json.loads(cleaned_response)
    except json.JSONDecodeError:
        return {
            "plain_english_explanation": raw_response,
            "practical_effect": (
                "Gemini returned an unstructured explanation. "
                "Review the explanation carefully."
            ),
            "main_risks": [],
            "questions_to_consider": [],
            "important_note": (
                "The response could not be converted into the "
                "expected structured format."
            ),
            "disclaimer": (
                "This explanation is general information and "
                "is not legal advice."
            ),
        }

    return {
        "plain_english_explanation": str(
            parsed.get(
                "plain_english_explanation",
                "No explanation was returned.",
            )
        ).strip(),
        "practical_effect": str(
            parsed.get(
                "practical_effect",
                "No practical effect was returned.",
            )
        ).strip(),
        "main_risks": _normalize_string_list(
            parsed.get("main_risks")
        ),
        "questions_to_consider": _normalize_string_list(
            parsed.get("questions_to_consider")
        ),
        "important_note": str(
            parsed.get(
                "important_note",
                "",
            )
        ).strip(),
        "disclaimer": (
            "This explanation is general information and "
            "is not legal advice."
        ),
    }
