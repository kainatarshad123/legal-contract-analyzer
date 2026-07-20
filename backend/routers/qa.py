"""Legal contract Q&A and clause-explanation routes."""

import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException, status

from db.database import (
    load_contract_chunks,
    load_contract_from_db,
    save_contract_chunks,
)
from schemas.qa import AskRequest, ExplainClauseRequest
from services.contract_store import CONTRACT_STORE
from services.gemini_client import (
    answer_contract_question_with_rag,
    explain_clause_with_gemini,
)
from services.rag_service import (
    build_contract_chunks,
    build_source_items,
    retrieve_relevant_chunks,
)
from services.structured_answers import build_structured_answer


logger = logging.getLogger(__name__)

router = APIRouter(tags=["legal-q-and-a"])

RAG_DISCLAIMER = (
    "This response is general information and not legal advice."
)


def load_saved_contract(
    contract_id: str,
) -> dict[str, Any]:
    """Load a contract from memory or SQLite."""

    contract_data = CONTRACT_STORE.get(contract_id)

    if contract_data is None:
        contract_data = load_contract_from_db(contract_id)

        if contract_data is not None:
            CONTRACT_STORE[contract_id] = contract_data

    if contract_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contract not found. Upload the contract again.",
        )

    return contract_data


def find_clause_by_number(
    contract_data: dict[str, Any],
    clause_number: int,
) -> dict[str, Any]:
    """Find one clause using its one-based clause number."""

    clauses = contract_data.get("clauses", [])

    if not isinstance(clauses, list):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The saved contract has invalid clause data.",
        )

    for clause in clauses:
        if not isinstance(clause, dict):
            continue

        stored_number = clause.get("clause_number")

        try:
            stored_number = int(stored_number)
        except (TypeError, ValueError):
            continue

        if stored_number == clause_number:
            return clause

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=(
            f"Clause {clause_number} was not found "
            "in this contract."
        ),
    )


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower()).strip()


def _should_use_structured_answer(question: str) -> bool:
    """Keep deterministic handling for supported overview questions."""

    normalized = _normalize_question(question)

    deterministic_patterns = (
        r"\bsummar(?:y|ize|ise)\b.*\bcontract\b",
        r"\bcontract\b.*\bsummar(?:y|ize|ise)\b",
        r"\brisky clauses?\b",
        r"\b(high|medium)\s+risk clauses?\b",
        r"\bwhat (information|details?|fields?) (is|are) missing\b",
        r"\bmissing (information|details?|fields?)\b",
        r"\bpayment terms?\b",
        r"\bhow much (rent|payment)\b",
        r"\btermination clause\b",
        r"\bhow can .* (agreement|contract|lease) be terminated\b",
    )

    return any(
        re.search(pattern, normalized)
        for pattern in deterministic_patterns
    )


def _get_or_create_chunks(
    contract_id: str,
    contract_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load existing chunks or backfill an older saved contract."""

    chunks = load_contract_chunks(contract_id)

    if chunks:
        return chunks

    chunks = build_contract_chunks(
        clauses=contract_data.get("clauses", []),
        contract_text=str(contract_data.get("text", "")),
    )

    if chunks:
        save_contract_chunks(
            contract_id=contract_id,
            chunks=chunks,
        )

    return chunks


@router.post("/ask-contract")
async def ask_contract(
    request: AskRequest,
) -> dict[str, Any]:
    """Answer a legal question using structured logic or RAG."""

    contract_data = load_saved_contract(
        request.contract_id
    )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    if _should_use_structured_answer(question):
        structured_result = build_structured_answer(
            contract_data,
            question,
        )

        if isinstance(structured_result, dict):
            structured_result.setdefault(
                "answer_type",
                "structured",
            )
            structured_result.setdefault(
                "sources",
                [],
            )

        return structured_result

    chunks = _get_or_create_chunks(
        contract_id=request.contract_id,
        contract_data=contract_data,
    )

    retrieved_chunks = retrieve_relevant_chunks(
        question=question,
        chunks=chunks,
        top_k=5,
        minimum_similarity=0.08,
    )

    sources = build_source_items(retrieved_chunks)

    try:
        generated = answer_contract_question_with_rag(
            question=question,
            retrieved_chunks=retrieved_chunks,
            filename=contract_data.get("filename"),
        )
    except ValueError as error:
        logger.warning(
            "Gemini configuration error: %s",
            error,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except Exception as error:
        logger.exception(
            "RAG contract answer failed.",
            exc_info=error,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "The contract question-answering service could "
                "not complete the request."
            ),
        ) from error

    return {
        "answer_type": "rag",
        "title": "Contract Question",
        "risk_level": "Unknown",
        "summary": generated.get("summary", ""),
        "answer": generated.get("answer", ""),
        "key_points": generated.get("key_points", []),
        "missing_fields": [],
        "related_clauses": generated.get(
            "related_clauses",
            [],
        ),
        "contract_details": {
            "retrieval_method": "tfidf_cosine_similarity",
            "retrieved_chunks": len(retrieved_chunks),
            "total_saved_chunks": len(chunks),
            "minimum_similarity": 0.08,
            "top_k": 5,
        },
        "sources": sources,
        "disclaimer": RAG_DISCLAIMER,
    }


@router.post("/explain-clause")
async def explain_clause(
    request: ExplainClauseRequest,
) -> dict[str, Any]:
    """Explain one contract clause in plain English."""

    contract_data = load_saved_contract(
        request.contract_id
    )

    clause = find_clause_by_number(
        contract_data=contract_data,
        clause_number=request.clause_number,
    )

    clause_text = str(
        clause.get("text")
        or clause.get("preview")
        or ""
    ).strip()

    if not clause_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The selected clause has no readable text.",
        )

    try:
        explanation = explain_clause_with_gemini(
            clause
        )
    except ValueError as error:
        logger.warning(
            "Gemini configuration error: %s",
            error,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Clause explanation failed.",
            exc_info=error,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "The clause explanation service could not "
                "complete the request."
            ),
        ) from error

    return {
        "contract_id": request.contract_id,
        "clause_number": request.clause_number,
        "clause_type": clause.get(
            "clause_type",
            "Not classified",
        ),
        "risk_level": clause.get(
            "risk_level",
            "Unknown",
        ),
        "party_affected": clause.get(
            "party_affected",
            "Not identified",
        ),
        "confidence_label": clause.get(
            "confidence_label",
            "Unknown",
        ),
        "needs_manual_review": bool(
            clause.get(
                "needs_manual_review",
                False,
            )
        ),
        **explanation,
    }
