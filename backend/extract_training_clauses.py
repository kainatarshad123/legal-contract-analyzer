from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import re

import fitz  # PyMuPDF
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "training_pdfs"
MODEL_DIR = BASE_DIR / "ml_model"

OUTPUT_FILE = MODEL_DIR / "phase5_clauses_for_labeling.csv"
REJECTED_FILE = MODEL_DIR / "phase5_rejected_fragments.csv"

MIN_WORDS = 10
MIN_CHARACTERS = 70
MAX_WORDS = 240

SYNTHETIC_FOOTER_TEXT = (
    "Synthetic lease document for ML training only - "
    "not legal advice or a live agreement."
)


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_inline_text(value: str) -> str:
    if not value:
        return ""

    value = value.replace("\x00", " ")
    value = value.replace("\u00ad", "")
    value = value.replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def clean_line(value: str) -> str:
    if not value:
        return ""

    value = value.replace("\x00", " ")
    value = value.replace("\u00ad", "")
    value = re.sub(r"[ \t]+", " ", value)

    return value.strip()


def normalize_for_duplicate_check(value: str) -> str:
    value = clean_inline_text(value).lower()
    value = re.sub(r"\[[^\]]*\]", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def create_clause_id(
    source: str,
    page_start: int,
    page_end: int,
    clause_text: str,
) -> str:
    raw = (
        f"{source}|{page_start}|{page_end}|"
        f"{normalize_for_duplicate_check(clause_text)}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def word_count(value: str) -> int:
    return len(clean_inline_text(value).split())


def appears_incomplete(value: str) -> bool:
    text = clean_inline_text(value)

    if not text:
        return True

    incomplete_endings = (
        "in accordance with",
        "subject to",
        "including but not limited to",
        "provided that",
        "as follows",
        "the following",
        "and",
        "or",
        "to",
        "of",
        "for",
        "with",
        "by",
    )

    lower_text = text.lower().rstrip(" ,;:-")

    if any(lower_text.endswith(item) for item in incomplete_endings):
        return True

    if text.count("(") > text.count(")"):
        return True

    return False


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_raw_pages(pdf_path: Path) -> list[dict]:
    pages = []

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            raw_text = page.get_text("text") or ""

            lines = [
                clean_line(line)
                for line in raw_text.splitlines()
                if clean_line(line)
            ]

            pages.append(
                {
                    "page_number": page_index + 1,
                    "lines": lines,
                }
            )

    return pages


def normalized_line_key(value: str) -> str:
    value = clean_line(value).lower()
    value = re.sub(r"\d+", "#", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def detect_repeated_page_lines(pages: list[dict]) -> set[str]:
    """
    Detect likely headers and footers repeated across several pages.

    Only the first three and last three lines of each page are considered.
    """
    counts = Counter()
    total_pages = len(pages)

    if total_pages < 2:
        return set()

    for page in pages:
        lines = page["lines"]

        candidates = lines[:3] + lines[-3:]

        for line in set(candidates):
            key = normalized_line_key(line)

            if len(key) >= 5:
                counts[key] += 1

    threshold = max(2, round(total_pages * 0.50))

    return {
        key
        for key, count in counts.items()
        if count >= threshold
    }


def is_page_artifact(
    line: str,
    repeated_page_lines: set[str],
) -> bool:
    cleaned = clean_line(line)
    lower_text = cleaned.lower()

    if not cleaned:
        return True

    if SYNTHETIC_FOOTER_TEXT.lower() in lower_text:
        return True

    if normalized_line_key(cleaned) in repeated_page_lines:
        return True

    page_patterns = [
        r"^page\s+\d+(?:\s+of\s+\d+)?$",
        r"^\d+\s*/\s*\d+$",
        r"^-\s*\d+\s*-$",
        r"^\d+$",
    ]

    if any(
        re.fullmatch(pattern, lower_text)
        for pattern in page_patterns
    ):
        return True

    return False


def clean_pages(pages: list[dict]) -> list[dict]:
    repeated_page_lines = detect_repeated_page_lines(pages)
    cleaned_pages = []

    for page in pages:
        cleaned_lines = [
            line
            for line in page["lines"]
            if not is_page_artifact(line, repeated_page_lines)
        ]

        cleaned_pages.append(
            {
                "page_number": page["page_number"],
                "lines": cleaned_lines,
                "text": "\n".join(cleaned_lines).strip(),
            }
        )

    return cleaned_pages


# ============================================================
# DOCUMENT ASSEMBLY
# ============================================================

def join_pages(pages: list[dict]) -> tuple[str, list[tuple[int, int, int]]]:
    """
    Join pages into one continuous document.

    Returns:
    - full document text
    - character ranges mapped to page numbers
    """
    parts = []
    page_ranges = []
    cursor = 0

    for page in pages:
        page_text = page["text"]

        if not page_text:
            continue

        if parts:
            separator = "\n"
            parts.append(separator)
            cursor += len(separator)

        start = cursor
        parts.append(page_text)
        cursor += len(page_text)
        end = cursor

        page_ranges.append(
            (start, end, page["page_number"])
        )

    return "".join(parts), page_ranges


def page_for_position(
    position: int,
    page_ranges: list[tuple[int, int, int]],
) -> int:
    for start, end, page_number in page_ranges:
        if start <= position <= end:
            return page_number

    return page_ranges[-1][2] if page_ranges else 1


# ============================================================
# CLAUSE SPLITTING
# ============================================================

CLAUSE_START_PATTERN = re.compile(
    r"""
    (?=
        (?:
            ^|\n
        )
        \s*
        (?:
            ARTICLE\s+[IVXLC\d]+
            |
            SECTION\s+\d+(?:\.\d+)*
            |
            \d+(?:\.\d+){0,5}[\.\)]
            |
            \(\d+(?:\.\d+){0,5}\)
            |
            \([a-z]\)
            |
            [a-z][\.\)]
            |
            \([ivxlcdm]+\)
            |
            [ivxlcdm]+[\.\)]
            |
            IN\s+WITNESS\s+WHEREOF
            |
            NOW\s+THIS\s+(?:DEED|AGREEMENT)\s+WITNESSETH
            |
            THE\s+SCHEDULE
            |
            SCHEDULE
            |
            EXECUTION
            |
            SIGNATURES?
        )
        \s+
    )
    """,
    flags=re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def add_clause_boundaries(text: str) -> str:
    """
    Normalize source line breaks and add boundaries before legal numbering.
    """
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Add boundaries where PDF extraction placed numbered clauses inline.
    inline_patterns = [
        r"(?<!\n)\s+(?=\d+(?:\.\d+){0,5}[\.\)]\s+[A-Z])",
        r"(?<!\n)\s+(?=\([a-z]\)\s+[A-Z])",
        r"(?<!\n)\s+(?=[a-z][\.\)]\s+[A-Z])",
        r"(?<!\n)\s+(?=\([ivxlcdm]+\)\s+[A-Z])",
        r"(?<!\n)\s+(?=(?:ARTICLE|SECTION)\s+[IVXLC\d]+)",
        r"(?<!\n)\s+(?=IN\s+WITNESS\s+WHEREOF)",
        r"(?<!\n)\s+(?=EXECUTION\b)",
        r"(?<!\n)\s+(?=SIGNATURES?\b)",
    ]

    for pattern in inline_patterns:
        text = re.sub(
            pattern,
            "\n",
            text,
            flags=re.IGNORECASE,
        )

    return text.strip()


def split_document_into_candidates(
    document_text: str,
    page_ranges: list[tuple[int, int, int]],
) -> list[dict]:
    text = add_clause_boundaries(document_text)

    matches = list(CLAUSE_START_PATTERN.finditer(text))

    if not matches:
        chunks = re.split(r"\n{2,}", text)
        output = []

        cursor = 0

        for chunk in chunks:
            chunk = clean_inline_text(chunk)

            if not chunk:
                continue

            start = text.find(chunk, cursor)
            end = start + len(chunk)
            cursor = end

            output.append(
                {
                    "text": chunk,
                    "page_start": page_for_position(start, page_ranges),
                    "page_end": page_for_position(end, page_ranges),
                }
            )

        return output

    candidates = []

    # Preserve usable preamble before the first numbered clause.
    first_start = matches[0].start()

    if first_start > 0:
        preamble = clean_inline_text(text[:first_start])

        if preamble:
            candidates.append(
                {
                    "text": preamble,
                    "page_start": page_for_position(0, page_ranges),
                    "page_end": page_for_position(
                        first_start,
                        page_ranges,
                    ),
                }
            )

    for index, match in enumerate(matches):
        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        candidate = clean_inline_text(text[start:end])

        if not candidate:
            continue

        candidates.append(
            {
                "text": candidate,
                "page_start": page_for_position(start, page_ranges),
                "page_end": page_for_position(end, page_ranges),
            }
        )

    return candidates


def split_oversized_candidate(candidate: dict) -> list[dict]:
    text = clean_inline_text(candidate["text"])

    if word_count(text) <= MAX_WORDS:
        return [candidate]

    # Prefer nested decimal subclauses such as 5.12.1, 5.12.2.
    parts = re.split(
        r"(?=\s+\d+(?:\.\d+){1,5}[\.\)]?\s+)",
        text,
    )

    parts = [
        clean_inline_text(part)
        for part in parts
        if clean_inline_text(part)
    ]

    if len(parts) <= 1:
        parts = re.split(
            r"(?=\s+(?:\([a-z]\)|[a-z][\.\)]|"
            r"\([ivxlcdm]+\)|[ivxlcdm]+[\.\)])\s+)",
            text,
            flags=re.IGNORECASE,
        )

        parts = [
            clean_inline_text(part)
            for part in parts
            if clean_inline_text(part)
        ]

    usable = [
        part
        for part in parts
        if word_count(part) >= MIN_WORDS
        and len(part) >= MIN_CHARACTERS
    ]

    if len(usable) <= 1:
        return [candidate]

    return [
        {
            "text": part,
            "page_start": candidate["page_start"],
            "page_end": candidate["page_end"],
        }
        for part in usable
    ]


def merge_incomplete_candidates(
    candidates: list[dict],
) -> list[dict]:
    """
    Merge short or incomplete fragments with their following clause.
    """
    merged = []
    index = 0

    while index < len(candidates):
        current = candidates[index].copy()
        current["text"] = clean_inline_text(current["text"])

        while (
            index + 1 < len(candidates)
            and (
                word_count(current["text"]) < MIN_WORDS
                or len(current["text"]) < MIN_CHARACTERS
                or appears_incomplete(current["text"])
            )
        ):
            following = candidates[index + 1]

            current["text"] = clean_inline_text(
                f"{current['text']} {following['text']}"
            )
            current["page_end"] = following["page_end"]

            index += 1

        merged.append(current)
        index += 1

    return merged


# ============================================================
# FILTERING
# ============================================================

def classify_fragment(clause_text: str) -> tuple[bool, str]:
    text = clean_inline_text(clause_text)
    count = word_count(text)

    if not text:
        return False, "empty"

    if SYNTHETIC_FOOTER_TEXT.lower() in text.lower():
        return False, "synthetic_footer"

    if re.search(r"\bpage\s+\d+(?:\s+of\s+\d+)?\b", text, re.I):
        return False, "page_label_inside_text"

    if len(text) < MIN_CHARACTERS:
        return False, "too_short_characters"

    if count < MIN_WORDS:
        return False, "too_few_words"

    if count > 500:
        return False, "extremely_long_mixed_clause"

    if len(re.findall(r"[a-zA-Z]", text)) < 25:
        return False, "not_enough_language"

    if appears_incomplete(text):
        return False, "appears_incomplete"

    return True, "accepted"


# ============================================================
# METADATA
# ============================================================

def infer_contract_type(filename: str) -> str:
    name = filename.lower()

    if "commercial" in name or "office" in name:
        return "Commercial Lease"

    if "month_to_month" in name or "month-to-month" in name:
        return "Month-to-Month Tenancy"

    if "short_term" in name or "short-term" in name:
        return "Short-Term Rental"

    if "formal" in name or "deed" in name:
        return "Formal Lease Deed"

    if "residential" in name or "tenancy" in name:
        return "Residential Lease"

    return "Lease Agreement"


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"Training PDF folder not found:\n{PDF_DIR}"
        )

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files were found in:\n{PDF_DIR}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    accepted_rows = []
    rejected_rows = []
    seen_clause_keys = set()

    print("TRAINING CLAUSE EXTRACTION")
    print("=" * 60)
    print(f"Input folder: {PDF_DIR}")
    print(f"PDF files found: {len(pdf_files)}")

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        try:
            raw_pages = extract_raw_pages(pdf_path)
            pages = clean_pages(raw_pages)
        except Exception as error:
            print(f"  ERROR: {error}")

            rejected_rows.append(
                {
                    "source": pdf_path.name,
                    "page_start": "",
                    "page_end": "",
                    "fragment_text": "",
                    "rejection_reason": f"pdf_read_error: {error}",
                }
            )
            continue

        empty_pages = [
            page["page_number"]
            for page in pages
            if not page["text"]
        ]

        document_text, page_ranges = join_pages(pages)

        if not document_text:
            rejected_rows.append(
                {
                    "source": pdf_path.name,
                    "page_start": "",
                    "page_end": "",
                    "fragment_text": "",
                    "rejection_reason": (
                        "no_selectable_text_possible_ocr"
                    ),
                }
            )

            print("  No selectable text found.")
            continue

        candidates = split_document_into_candidates(
            document_text,
            page_ranges,
        )

        candidates = merge_incomplete_candidates(candidates)

        expanded_candidates = []

        for candidate in candidates:
            expanded_candidates.extend(
                split_oversized_candidate(candidate)
            )

        document_accepted = 0
        document_rejected = 0

        for candidate in expanded_candidates:
            text = clean_inline_text(candidate["text"])
            page_start = int(candidate["page_start"])
            page_end = int(candidate["page_end"])

            keep, reason = classify_fragment(text)

            if not keep:
                rejected_rows.append(
                    {
                        "source": pdf_path.name,
                        "page_start": page_start,
                        "page_end": page_end,
                        "fragment_text": text,
                        "rejection_reason": reason,
                    }
                )
                document_rejected += 1
                continue

            duplicate_key = normalize_for_duplicate_check(text)

            if duplicate_key in seen_clause_keys:
                rejected_rows.append(
                    {
                        "source": pdf_path.name,
                        "page_start": page_start,
                        "page_end": page_end,
                        "fragment_text": text,
                        "rejection_reason": "duplicate",
                    }
                )
                document_rejected += 1
                continue

            seen_clause_keys.add(duplicate_key)

            accepted_rows.append(
                {
                    "clause_id": create_clause_id(
                        pdf_path.name,
                        page_start,
                        page_end,
                        text,
                    ),
                    "clause_text": text,
                    "clause_type": "",
                    "source": pdf_path.name,
                    "contract_type": infer_contract_type(
                        pdf_path.name
                    ),
                    "page_start": page_start,
                    "page_end": page_end,
                    "word_count": word_count(text),
                    "character_count": len(text),
                    "label_status": "unlabeled",
                    "review_notes": "",
                    "source_split": "train",
                    "dataset_origin": "Phase 5 Lease PDFs",
                }
            )

            document_accepted += 1

        print(f"  Pages: {len(pages)}")
        print(
            "  Pages without usable selectable text: "
            f"{len(empty_pages)}"
        )
        print(f"  Accepted clauses: {document_accepted}")
        print(f"  Rejected fragments: {document_rejected}")

    accepted_df = pd.DataFrame(
        accepted_rows,
        columns=[
            "clause_id",
            "clause_text",
            "clause_type",
            "source",
            "contract_type",
            "page_start",
            "page_end",
            "word_count",
            "character_count",
            "label_status",
            "review_notes",
            "source_split",
            "dataset_origin",
        ],
    )

    rejected_df = pd.DataFrame(
        rejected_rows,
        columns=[
            "source",
            "page_start",
            "page_end",
            "fragment_text",
            "rejection_reason",
        ],
    )

    accepted_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    rejected_df.to_csv(
        REJECTED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Accepted clauses: {len(accepted_df)}")
    print(f"Rejected fragments: {len(rejected_df)}")
    print(
        "Unique source files: "
        f"{accepted_df['source'].nunique() if not accepted_df.empty else 0}"
    )

    if not accepted_df.empty:
        print("\nAccepted clauses by source:")
        print(
            accepted_df["source"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print("\nClause-length summary:")
        print(
            accepted_df["word_count"]
            .describe()
            .round(2)
            .to_string()
        )

        print(
            "\nClauses longer than 250 words: "
            f"{(accepted_df['word_count'] > 250).sum()}"
        )

        print(
            "Clauses shorter than 15 words: "
            f"{(accepted_df['word_count'] < 15).sum()}"
        )

    print(f"\nLabeling file saved to:\n{OUTPUT_FILE}")
    print(f"\nRejected-fragment review saved to:\n{REJECTED_FILE}")
    print(
        "\nNext: send both regenerated CSV files for review "
        "before labeling."
    )


if __name__ == "__main__":
    main()