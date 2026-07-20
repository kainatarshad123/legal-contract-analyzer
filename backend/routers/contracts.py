"""Contract upload, history, retrieval, and deletion routes."""

import uuid
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from db.database import (
    delete_contract_from_db,
    list_contracts_from_db,
    load_contract_from_db,
    save_contract_to_db,
)
from services.basic_answers import build_summary_answer
from services.contract_analysis import (
    analyze_clauses,
    extract_contract_type,
    extract_missing_fields,
    get_overall_risk,
)
from services.contract_store import CONTRACT_STORE
from services.pdf_extraction import extract_pdf_content
from services.text_processing import (
    normalize_raw_text,
    remove_irrelevant_sections,
    smart_preview,
    split_into_clauses,
)
from services.upload_validator import validate_pdf_upload


router = APIRouter(tags=["contracts"])


@router.post("/upload-contract")
async def upload_contract(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    file_bytes = await validate_pdf_upload(file)

    extraction_result = extract_pdf_content(file_bytes)
    raw_text = str(extraction_result.get("text", ""))
    normalized_text = normalize_raw_text(raw_text)
    cleaned_contract_text = remove_irrelevant_sections(normalized_text)

    if not cleaned_contract_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No readable contract text could be extracted from the PDF.",
        )

    raw_clauses = split_into_clauses(cleaned_contract_text)

    if not raw_clauses:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No analyzable contract clauses were found in the PDF.",
        )

    analyzed_clauses = analyze_clauses(raw_clauses)
    contract_id = str(uuid.uuid4())
    overall_risk = get_overall_risk(analyzed_clauses)

    contract_data = {
        "text": cleaned_contract_text,
        "clauses": analyzed_clauses,
    }

    confidence_counts = {
        "High": sum(
            clause.get("confidence_label") == "High"
            for clause in analyzed_clauses
        ),
        "Medium": sum(
            clause.get("confidence_label") == "Medium"
            for clause in analyzed_clauses
        ),
        "Low": sum(
            clause.get("confidence_label") == "Low"
            for clause in analyzed_clauses
        ),
    }

    manual_review_count = sum(
        bool(clause.get("needs_manual_review"))
        for clause in analyzed_clauses
    )

    analysis = {
        "contract_type": extract_contract_type(cleaned_contract_text),
        "overall_risk": overall_risk,
        "total_clauses": len(analyzed_clauses),
        "risky_clauses": sum(
            clause["risk_level"] in {"Medium", "High"}
            for clause in analyzed_clauses
        ),
        "manual_review_clauses": manual_review_count,
        "high_confidence_clauses": confidence_counts["High"],
        "medium_confidence_clauses": confidence_counts["Medium"],
        "low_confidence_clauses": confidence_counts["Low"],
        "confidence_summary": confidence_counts,
        "missing_fields": extract_missing_fields(cleaned_contract_text),
        "summary": build_summary_answer(contract_data),
    }

    stored_contract = {
        "filename": file.filename,
        "content_type": file.content_type,
        "text": cleaned_contract_text,
        "clauses": analyzed_clauses,
        "analysis": analysis,
    }
    CONTRACT_STORE[contract_id] = stored_contract

    save_contract_to_db(
        contract_id=contract_id,
        filename=file.filename,
        content_type=file.content_type,
        contract_text=cleaned_contract_text,
        clauses=analyzed_clauses,
        analysis=analysis,
    )

    return {
        "contract_id": contract_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "Contract uploaded, analyzed, and saved successfully.",
        "raw_total_characters": len(raw_text),
        "total_characters": len(cleaned_contract_text),
        "page_count": extraction_result["page_count"],
        "extraction_method": extraction_result["extraction_method"],
        "text_pages": extraction_result["text_pages"],
        "ocr_pages": extraction_result["ocr_pages"],
        "failed_ocr_pages": extraction_result["failed_ocr_pages"],
        "text_preview": smart_preview(
            cleaned_contract_text,
            max_length=700,
        ),
        "analysis": analysis,
        "clauses": analyzed_clauses,
    }


@router.get("/contracts")
def get_contracts() -> dict[str, list[dict[str, Any]]]:
    return {"contracts": list_contracts_from_db()}


@router.get("/contracts/{contract_id}")
def get_contract_by_id(contract_id: str) -> dict[str, Any]:
    contract_data = load_contract_from_db(contract_id)

    if contract_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contract not found.",
        )

    return {
        "error": False,
        "contract_id": contract_id,
        "filename": contract_data.get("filename"),
        "message": "Contract loaded successfully.",
        "analysis": contract_data.get("analysis"),
        "clauses": contract_data.get("clauses"),
    }


@router.delete("/contracts/{contract_id}")
def delete_contract(contract_id: str) -> dict[str, Any]:
    deleted = delete_contract_from_db(contract_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contract not found.",
        )

    CONTRACT_STORE.pop(contract_id, None)

    return {
        "error": False,
        "message": "Contract deleted successfully.",
        "contract_id": contract_id,
    }