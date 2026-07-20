"""Clause-aware TF-IDF retrieval for contract question answering."""

import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_TOP_K = 5
DEFAULT_MINIMUM_SIMILARITY = 0.08
DEFAULT_MAX_CHUNK_CHARACTERS = 1800
DEFAULT_OVERLAP_CHARACTERS = 250


def _clean_text(text: Any) -> str:
    """Normalize whitespace without altering legal wording."""

    return re.sub(r"\s+", " ", str(text or "")).strip()


def _split_long_text(
    text: str,
    max_characters: int,
    overlap_characters: int,
) -> list[str]:
    """Split long text into readable overlapping windows."""

    cleaned = _clean_text(text)

    if not cleaned:
        return []

    if len(cleaned) <= max_characters:
        return [cleaned]

    sentence_parts = re.split(
        r"(?<=[.!?;:])\s+(?=[A-Z0-9(])",
        cleaned,
    )

    pieces: list[str] = []
    current = ""

    for sentence in sentence_parts:
        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = (
            f"{current} {sentence}".strip()
            if current
            else sentence
        )

        if len(candidate) <= max_characters:
            current = candidate
            continue

        if current:
            pieces.append(current)

            overlap = current[-overlap_characters:].strip()
            current = (
                f"{overlap} {sentence}".strip()
                if overlap
                else sentence
            )
        else:
            start = 0
            step = max(1, max_characters - overlap_characters)

            while start < len(sentence):
                pieces.append(
                    sentence[start:start + max_characters].strip()
                )
                start += step

            current = ""

    if current:
        pieces.append(current)

    return [piece for piece in pieces if piece]


def build_contract_chunks(
    clauses: list[dict[str, Any]] | None,
    contract_text: str,
    max_chunk_characters: int = DEFAULT_MAX_CHUNK_CHARACTERS,
    overlap_characters: int = DEFAULT_OVERLAP_CHARACTERS,
) -> list[dict[str, Any]]:
    """Build retrieval chunks, preferring analyzed clause boundaries."""

    chunks: list[dict[str, Any]] = []
    safe_clauses = clauses if isinstance(clauses, list) else []

    for position, clause in enumerate(safe_clauses, start=1):
        if not isinstance(clause, dict):
            continue

        clause_text = _clean_text(
            clause.get("text")
            or clause.get("preview")
            or clause.get("clause_text")
        )

        if not clause_text:
            continue

        raw_number = clause.get("clause_number", position)

        try:
            clause_number = int(raw_number)
        except (TypeError, ValueError):
            clause_number = position

        clause_type = str(
            clause.get("clause_type")
            or "Not classified"
        ).strip()

        clause_parts = _split_long_text(
            clause_text,
            max_characters=max_chunk_characters,
            overlap_characters=overlap_characters,
        )

        for part in clause_parts:
            chunks.append(
                {
                    "chunk_index": len(chunks),
                    "clause_number": clause_number,
                    "clause_type": clause_type,
                    "text": part,
                }
            )

    if chunks:
        return chunks

    # Fallback for old or malformed records without analyzed clauses.
    full_text_parts = _split_long_text(
        _clean_text(contract_text),
        max_characters=max_chunk_characters,
        overlap_characters=overlap_characters,
    )

    return [
        {
            "chunk_index": index,
            "clause_number": None,
            "clause_type": "Contract text",
            "text": part,
        }
        for index, part in enumerate(full_text_parts)
    ]


def retrieve_relevant_chunks(
    question: str,
    chunks: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    minimum_similarity: float = DEFAULT_MINIMUM_SIMILARITY,
) -> list[dict[str, Any]]:
    """Rank contract chunks against a question using TF-IDF cosine similarity."""

    cleaned_question = _clean_text(question)

    if not cleaned_question or not chunks:
        return []

    valid_chunks = [
        chunk
        for chunk in chunks
        if _clean_text(chunk.get("text") or chunk.get("chunk_text"))
    ]

    if not valid_chunks:
        return []

    corpus = [
        _clean_text(chunk.get("text") or chunk.get("chunk_text"))
        for chunk in valid_chunks
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    try:
        matrix = vectorizer.fit_transform(corpus + [cleaned_question])
    except ValueError:
        return []

    chunk_matrix = matrix[:-1]
    question_vector = matrix[-1]
    scores = cosine_similarity(
        question_vector,
        chunk_matrix,
    ).flatten()

    ranked_indices = scores.argsort()[::-1]
    results: list[dict[str, Any]] = []

    for index in ranked_indices:
        similarity = float(scores[index])

        if similarity < minimum_similarity:
            continue

        item = dict(valid_chunks[int(index)])
        item["text"] = corpus[int(index)]
        item["similarity"] = round(similarity, 4)
        results.append(item)

        if len(results) >= max(1, top_k):
            break

    return results


def format_rag_context(
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """Format retrieved chunks for a grounded Gemini prompt."""

    sections: list[str] = []

    for source_number, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        clause_number = chunk.get("clause_number")
        clause_type = (
            chunk.get("clause_type")
            or "Not classified"
        )
        similarity = float(chunk.get("similarity", 0.0))
        text = _clean_text(
            chunk.get("text")
            or chunk.get("chunk_text")
        )

        sections.append(
            "\n".join(
                [
                    f"[SOURCE {source_number}]",
                    f"Clause number: {clause_number or 'Unknown'}",
                    f"Clause type: {clause_type}",
                    f"Retrieval similarity: {similarity:.4f}",
                    f"Text: {text}",
                ]
            )
        )

    return "\n\n".join(sections)


def _snippet(text: str, max_length: int = 360) -> str:
    """Create a compact source preview."""

    cleaned = _clean_text(text)

    if len(cleaned) <= max_length:
        return cleaned

    return cleaned[: max_length - 1].rstrip() + "…"


def build_source_items(
    retrieved_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create frontend-ready source metadata."""

    return [
        {
            "source_number": source_number,
            "chunk_index": chunk.get("chunk_index"),
            "clause_number": chunk.get("clause_number"),
            "clause_type": (
                chunk.get("clause_type")
                or "Not classified"
            ),
            "similarity": round(
                float(chunk.get("similarity", 0.0)),
                4,
            ),
            "snippet": _snippet(
                str(
                    chunk.get("text")
                    or chunk.get("chunk_text")
                    or ""
                )
            ),
        }
        for source_number, chunk in enumerate(
            retrieved_chunks,
            start=1,
        )
    ]
