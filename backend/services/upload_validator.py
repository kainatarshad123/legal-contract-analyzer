"""Security validation for uploaded PDF contracts."""

import os

import fitz
from fastapi import HTTPException, UploadFile, status


DEFAULT_MAX_FILE_SIZE_MB = 15
DEFAULT_MAX_PAGE_COUNT = 100
READ_CHUNK_SIZE = 1024 * 1024
ALLOWED_PDF_MIME_TYPES = {
    "application/pdf",
    "application/x-pdf",
}


def _positive_int_from_env(name: str, default: int) -> int:
    """Read a positive integer setting, falling back to a safe default."""
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError:
        print(f"Invalid {name} value {raw_value!r}; using {default}.")
        return default

    if value <= 0:
        print(f"{name} must be positive; using {default}.")
        return default

    return value


MAX_FILE_SIZE_MB = _positive_int_from_env(
    "MAX_PDF_FILE_SIZE_MB",
    DEFAULT_MAX_FILE_SIZE_MB,
)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_PAGE_COUNT = _positive_int_from_env(
    "MAX_PDF_PAGE_COUNT",
    DEFAULT_MAX_PAGE_COUNT,
)


async def _read_upload_with_size_limit(file: UploadFile) -> bytes:
    """Read an upload incrementally and stop once the size limit is exceeded."""
    chunks: list[bytes] = []
    total_size = 0

    while True:
        chunk = await file.read(READ_CHUNK_SIZE)

        if not chunk:
            break

        total_size += len(chunk)

        if total_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"PDF exceeds the maximum allowed size of "
                    f"{MAX_FILE_SIZE_MB} MB."
                ),
            )

        chunks.append(chunk)

    file_bytes = b"".join(chunks)

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    return file_bytes


def _validate_pdf_mime_type(file: UploadFile) -> None:
    content_type = (file.content_type or "").lower().strip()

    if content_type not in ALLOWED_PDF_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )


def _validate_pdf_signature(file_bytes: bytes) -> None:
    if not file_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="The uploaded file does not have a valid PDF signature.",
        )


def _validate_pdf_structure_and_page_count(file_bytes: bytes) -> int:
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded PDF is corrupted or cannot be opened.",
        ) from error

    try:
        page_count = document.page_count

        if page_count <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded PDF contains no pages.",
            )

        if page_count > MAX_PAGE_COUNT:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"PDF contains {page_count} pages. "
                    f"The maximum allowed page count is {MAX_PAGE_COUNT}."
                ),
            )

        return page_count
    finally:
        document.close()


async def validate_pdf_upload(file: UploadFile) -> bytes:
    """Validate MIME type, size, PDF signature, structure, and page count.

    This function must run before PDF extraction or OCR.
    """
    _validate_pdf_mime_type(file)
    file_bytes = await _read_upload_with_size_limit(file)
    _validate_pdf_signature(file_bytes)
    _validate_pdf_structure_and_page_count(file_bytes)

    return file_bytes