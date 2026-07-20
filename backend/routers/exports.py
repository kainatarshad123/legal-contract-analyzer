"""Routes for exporting contract-analysis reports."""

import io
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from db.database import load_contract_from_db
from services.report_export import (
    build_docx_report,
    build_pdf_report,
    sanitize_filename,
)


router = APIRouter(
    prefix="/contracts",
    tags=["report-exports"],
)


def load_export_contract(
    contract_id: str,
) -> dict[str, Any]:
    """Load a saved contract for report generation."""

    contract_data = load_contract_from_db(contract_id)

    if contract_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contract not found.",
        )

    return contract_data


@router.get("/{contract_id}/export/pdf")
def export_contract_pdf(
    contract_id: str,
) -> StreamingResponse:
    """Download the contract analysis as a PDF report."""

    contract_data = load_export_contract(contract_id)
    report_bytes = build_pdf_report(contract_data)

    safe_name = sanitize_filename(
        contract_data.get("filename")
    )

    headers = {
        "Content-Disposition": (
            f'attachment; filename="{safe_name}_analysis.pdf"'
        )
    }

    return StreamingResponse(
        io.BytesIO(report_bytes),
        media_type="application/pdf",
        headers=headers,
    )


@router.get("/{contract_id}/export/docx")
def export_contract_docx(
    contract_id: str,
) -> StreamingResponse:
    """Download the contract analysis as a DOCX report."""

    contract_data = load_export_contract(contract_id)
    report_bytes = build_docx_report(contract_data)

    safe_name = sanitize_filename(
        contract_data.get("filename")
    )

    headers = {
        "Content-Disposition": (
            f'attachment; filename="{safe_name}_analysis.docx"'
        )
    }

    return StreamingResponse(
        io.BytesIO(report_bytes),
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        headers=headers,
    )