"""Text cleaning and clause segmentation utilities."""

import re


def normalize_raw_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def remove_irrelevant_sections(text: str) -> str:
    """
    Removes legal website/template explanation sections and keeps the actual contract draft.
    """

    if not text:
        return ""

    start_patterns = [
        r"Draft of Deed of Lease",
        r"This Deed of Lease is made",
        r"THIS DEED OF LEASE",
        r"NOW This Deed Witnesseth",
        r"NOW THIS DEED WITNESSETH",
    ]

    start_index = None

    for pattern in start_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            start_index = match.start()
            break

    if start_index is not None:
        text = text[start_index:]

    stop_patterns = [
        r"Documents Required",
        r"Procedure for",
        r"Legal Considerations",
        r"How can a lawyer help",
        r"What is Deed of Family Trust",
        r"What is Lease Deed",
        r"Why is Lease Deed",
    ]

    stop_indexes = []

    for pattern in stop_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            stop_indexes.append(match.start())

    if stop_indexes:
        text = text[:min(stop_indexes)]

    return text.strip()


def clean_display_text(text: str) -> str:
    """
    Cleans text for frontend display.
    This improves readability without changing legal meaning.
    """

    if not text:
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    replacements = {
        "convenants": "covenants",
        "If the-ground": "If the ground",
        "if the-ground": "if the ground",
        "an arrears": "in arrears",
        "reas...": "reason",
    }

    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    # Replace visible blanks with cleaner wording
    text = re.sub(r"\.{3,}", " [not filled in] ", text)
    text = re.sub(r"…+", " [not filled in] ", text)
    text = text.replace("20___", "[year not filled in]")
    text = re.sub(r"_{2,}", " [not filled in] ", text)

    # Clean duplicate placeholder wording
    text = re.sub(r"(\[not filled in\]\s*){2,}", "[not filled in] ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def smart_preview(text: str, max_length: int = 420) -> str:
    """
    Creates a clean frontend preview without cutting words badly.
    """

    text = clean_display_text(text)

    if len(text) <= max_length:
        return text

    cut = text[:max_length]

    # Prefer ending at a sentence boundary.
    sentence_end = max(
        cut.rfind(". "),
        cut.rfind("; "),
        cut.rfind(": ")
    )

    if sentence_end > 180:
        return cut[:sentence_end + 1].strip()

    # Otherwise end at a comma if available.
    comma_end = cut.rfind(", ")

    if comma_end > 220:
        return cut[:comma_end + 1].strip()

    # Otherwise end at the last complete word.
    last_space = cut.rfind(" ")

    if last_space > 180:
        return cut[:last_space].strip() + "..."

    return cut.strip() + "..."


def has_blank_or_placeholder(text: str) -> bool:
    if not text:
        return True

    patterns = [
        "[not filled in]",
        "[year not filled in]",
        "[blank]",
        "[year blank]",
        "....",
        "...",
        "___",
        "_____",
        "20___",
    ]

    lower_text = text.lower()

    return any(pattern.lower() in lower_text for pattern in patterns)


def value_or_not_specified(value: str) -> str:
    if not value or has_blank_or_placeholder(value):
        return "Not specified in the document."
    return clean_display_text(value)


def split_into_clauses(text: str) -> list[str]:
    """
    Splits contract text into meaningful legal clauses/sections.

    This version avoids splitting on every 'The Lessor' or 'The Lessee'
    because that was creating broken fragments.
    """

    text = normalize_raw_text(text)

    # Convert excessive newlines into single spaces first.
    text = re.sub(r"\s+", " ", text).strip()

    # Add split markers before major numbered clauses only.
    # Example: 1. In pursuance..., 2. The Lessee..., 3. The Lessor...
    text = re.sub(
        r"\s(?=(?:\d+)\.\s)",
        "\n",
        text
    )

    # Add split markers before major legal sections.
    major_section_patterns = [
        r"NOW\s+This\s+Deed\s+Witnesseth",
        r"NOW\s+THIS\s+DEED\s+WITNESSETH",
        r"IN\s+WITNESS\s+WHEREOF",
        r"In\s+witness\s+whereof",
        r"THE\s+SCHEDULE\s+ABOVE\s+REFERRED\s+TO",
        r"The\s+Schedule\s+Above\s+Referred\s+To",
    ]

    for pattern in major_section_patterns:
        text = re.sub(
            rf"\s(?=({pattern}))",
            "\n",
            text,
            flags=re.IGNORECASE
        )

    raw_sections = [section.strip() for section in text.split("\n") if section.strip()]

    clauses = []

    for section in raw_sections:
        section = clean_display_text(section)

        if len(section) < 60:
            continue

        # If a numbered section contains lettered subclauses, keep them together
        # unless the section is extremely long.
        if len(section) > 2500:
            subclauses = split_large_clause_by_lettered_parts(section)
            clauses.extend(subclauses)
        else:
            clauses.append(section)

    # Final cleanup: remove duplicates and tiny broken fragments
    cleaned_clauses = []
    seen = set()

    for clause in clauses:
        clause = clean_display_text(clause)

        if len(clause) < 60:
            continue

        # Avoid fragments that are clearly not useful standalone clauses
        bad_endings = [
            "hereinafter called '",
            "in the manner hereinafter appearin",
            "regularl",
            "notic",
            "withhel",
        ]

        if any(clause.lower().endswith(item.lower()) for item in bad_endings):
            continue

        key = clause[:120].lower()

        if key not in seen:
            cleaned_clauses.append(clause)
            seen.add(key)

    return cleaned_clauses


def split_large_clause_by_lettered_parts(section: str) -> list[str]:
    """
    Splits very large numbered clauses into lettered subclauses only when needed.
    Example: a. To pay rent...
             b. To bear taxes...
             c. To maintain premises...
    """

    section = clean_display_text(section)

    # Split only before real lettered subclauses like "a. ", "b. ", "c. "
    parts = re.split(r"(?=\s[a-h]\.\s)", section)

    cleaned_parts = []

    for part in parts:
        part = clean_display_text(part)

        if len(part) < 80:
            continue

        cleaned_parts.append(part)

    if cleaned_parts:
        return cleaned_parts

    return [section]