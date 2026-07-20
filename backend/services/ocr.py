"""OCR helpers for scanned PDF pages."""

from typing import Any

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None  # type: ignore[assignment]


def is_ocr_available() -> bool:
    """Return True when both pytesseract and Pillow are installed."""
    return pytesseract is not None and Image is not None


def extract_text_from_pixmap(
    pixmap: Any,
    *,
    config: str = "--oem 3 --psm 6",
) -> str:
    """Run Tesseract OCR on a PyMuPDF pixmap."""
    if not is_ocr_available():
        raise RuntimeError(
            "OCR dependencies are unavailable. Install pytesseract and Pillow."
        )

    image = Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )

    return pytesseract.image_to_string(
        image,
        config=config,
    ).strip()