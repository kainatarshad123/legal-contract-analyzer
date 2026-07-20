"""PDF text extraction with page-level OCR fallback."""

import re
from typing import Any

import fitz

from services.ocr import extract_text_from_pixmap, is_ocr_available


MIN_SELECTABLE_TEXT_CHARACTERS = 40
OCR_ZOOM = 2.0


def extract_pdf_content(file_bytes: bytes) -> dict[str, Any]:
    """Extract selectable text and OCR pages containing insufficient text."""
    pdf = fitz.open(stream=file_bytes, filetype="pdf")

    extracted_pages: list[str] = []
    text_pages = 0
    ocr_pages = 0
    failed_ocr_pages: list[int] = []

    try:
        for page_number, page in enumerate(pdf, start=1):
            page_text = (page.get_text("text") or "").strip()
            compact_text = re.sub(r"\s+", "", page_text)

            if len(compact_text) >= MIN_SELECTABLE_TEXT_CHARACTERS:
                extracted_pages.append(page_text)
                text_pages += 1
                continue

            if not is_ocr_available():
                failed_ocr_pages.append(page_number)
                extracted_pages.append(page_text)
                continue

            try:
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(OCR_ZOOM, OCR_ZOOM),
                    alpha=False,
                )
                ocr_text = extract_text_from_pixmap(pixmap)
                extracted_pages.append(ocr_text)
                ocr_pages += 1
            except Exception as error:
                print(f"OCR failed on page {page_number}: {error}")
                failed_ocr_pages.append(page_number)
                extracted_pages.append(page_text)
    finally:
        pdf.close()

    if text_pages > 0 and ocr_pages > 0:
        extraction_method = "mixed"
    elif ocr_pages > 0:
        extraction_method = "ocr"
    else:
        extraction_method = "text"

    return {
        "text": "\n\n".join(extracted_pages).strip(),
        "page_count": len(extracted_pages),
        "text_pages": text_pages,
        "ocr_pages": ocr_pages,
        "failed_ocr_pages": failed_ocr_pages,
        "extraction_method": extraction_method,
    }


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Return only the extracted text from a PDF."""
    extracted_text = extract_pdf_content(file_bytes).get("text", "")
    return str(extracted_text)