"""Compare two analyzed contracts at clause level."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


RISK_ORDER = {
    "Unknown": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
}


def normalize_text(value: Any) -> str:
    """Normalize clause text before comparison."""

    if value is None:
        return ""

    return " ".join(str(value).lower().split())


def get_clause_text(clause: dict[str, Any]) -> str:
    """Return clause text from supported clause fields."""

    return str(
        clause.get("text")
        or clause.get("clause_text")
        or clause.get("preview")
        or ""
    ).strip()


def get_clause_type(clause: dict[str, Any]) -> str:
    """Return a readable clause type."""

    return str(
        clause.get("clause_type")
        or clause.get("type")
        or "Unclassified"
    ).strip()


def get_risk_level(clause: dict[str, Any]) -> str:
    """Return normalized risk level."""

    risk = str(
        clause.get("risk_level")
        or clause.get("risk")
        or "Unknown"
    ).strip().title()

    if risk not in RISK_ORDER:
        return "Unknown"

    return risk


def get_confidence(clause: dict[str, Any]) -> float:
    """Return confidence as a float between 0 and 1."""

    raw_value = (
        clause.get("clause_type_confidence")
        or clause.get("confidence")
        or 0.0
    )

    try:
        confidence = float(raw_value)
    except (TypeError, ValueError):
        confidence = 0.0

    return max(0.0, min(confidence, 1.0))


def calculate_similarity(
    first_clause: dict[str, Any],
    second_clause: dict[str, Any],
) -> float:
    """Calculate similarity between two clauses."""

    first_text = normalize_text(get_clause_text(first_clause))
    second_text = normalize_text(get_clause_text(second_clause))

    if not first_text or not second_text:
        return 0.0

    text_similarity = SequenceMatcher(
        None,
        first_text,
        second_text,
    ).ratio()

    first_type = get_clause_type(first_clause).lower()
    second_type = get_clause_type(second_clause).lower()

    type_bonus = 0.12 if first_type == second_type else 0.0

    return min(text_similarity + type_bonus, 1.0)


def determine_risk_change(
    old_risk: str,
    new_risk: str,
) -> str:
    """Return whether risk increased, decreased, or stayed unchanged."""

    old_value = RISK_ORDER.get(old_risk, 0)
    new_value = RISK_ORDER.get(new_risk, 0)

    if new_value > old_value:
        return "increased"

    if new_value < old_value:
        return "decreased"

    return "unchanged"


def find_best_match(
    base_clause: dict[str, Any],
    comparison_clauses: list[dict[str, Any]],
    used_indexes: set[int],
    minimum_similarity: float = 0.48,
) -> tuple[int | None, float]:
    """Find the best unused comparison clause."""

    best_index: int | None = None
    best_similarity = 0.0

    for index, comparison_clause in enumerate(comparison_clauses):
        if index in used_indexes:
            continue

        similarity = calculate_similarity(
            base_clause,
            comparison_clause,
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_index = index

    if best_similarity < minimum_similarity:
        return None, best_similarity

    return best_index, best_similarity


def build_clause_snapshot(
    clause: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a consistent clause representation."""

    if clause is None:
        return None

    return {
        "clause_number": clause.get("clause_number"),
        "clause_type": get_clause_type(clause),
        "text": get_clause_text(clause),
        "risk_level": get_risk_level(clause),
        "confidence": get_confidence(clause),
        "confidence_label": clause.get("confidence_label"),
        "needs_manual_review": bool(
            clause.get("needs_manual_review", False)
        ),
        "review_status": clause.get("review_status"),
        "party_affected": (
            clause.get("party_affected")
            or clause.get("affected_party")
        ),
        "risk_reason": (
            clause.get("risk_reason")
            or clause.get("reason")
        ),
        "recommended_action": (
            clause.get("recommended_action")
            or clause.get("action")
        ),
    }


def compare_contracts(
    base_contract: dict[str, Any],
    comparison_contract: dict[str, Any],
) -> dict[str, Any]:
    """Compare two saved contracts and return clause-level changes."""

    base_clauses = base_contract.get("clauses", [])
    comparison_clauses = comparison_contract.get("clauses", [])

    if not isinstance(base_clauses, list):
        base_clauses = []

    if not isinstance(comparison_clauses, list):
        comparison_clauses = []

    base_clauses = [
        clause
        for clause in base_clauses
        if isinstance(clause, dict)
    ]

    comparison_clauses = [
        clause
        for clause in comparison_clauses
        if isinstance(clause, dict)
    ]

    used_comparison_indexes: set[int] = set()
    clause_changes: list[dict[str, Any]] = []

    summary = {
        "added": 0,
        "removed": 0,
        "changed": 0,
        "unchanged": 0,
        "risk_increased": 0,
        "risk_decreased": 0,
        "manual_review_added": 0,
        "manual_review_removed": 0,
    }

    for base_index, base_clause in enumerate(base_clauses):
        match_index, similarity = find_best_match(
            base_clause,
            comparison_clauses,
            used_comparison_indexes,
        )

        if match_index is None:
            summary["removed"] += 1

            clause_changes.append(
                {
                    "change_type": "removed",
                    "similarity": round(similarity, 4),
                    "base_clause": build_clause_snapshot(base_clause),
                    "comparison_clause": None,
                    "risk_change": "removed",
                    "confidence_change": None,
                    "manual_review_change": "removed"
                    if base_clause.get("needs_manual_review")
                    else "unchanged",
                }
            )

            continue

        used_comparison_indexes.add(match_index)
        comparison_clause = comparison_clauses[match_index]

        old_text = normalize_text(get_clause_text(base_clause))
        new_text = normalize_text(get_clause_text(comparison_clause))

        old_type = get_clause_type(base_clause)
        new_type = get_clause_type(comparison_clause)

        old_risk = get_risk_level(base_clause)
        new_risk = get_risk_level(comparison_clause)

        old_confidence = get_confidence(base_clause)
        new_confidence = get_confidence(comparison_clause)

        old_review = bool(
            base_clause.get("needs_manual_review", False)
        )
        new_review = bool(
            comparison_clause.get("needs_manual_review", False)
        )

        risk_change = determine_risk_change(
            old_risk,
            new_risk,
        )

        if risk_change == "increased":
            summary["risk_increased"] += 1
        elif risk_change == "decreased":
            summary["risk_decreased"] += 1

        if not old_review and new_review:
            manual_review_change = "added"
            summary["manual_review_added"] += 1
        elif old_review and not new_review:
            manual_review_change = "removed"
            summary["manual_review_removed"] += 1
        else:
            manual_review_change = "unchanged"

        is_unchanged = (
            similarity >= 0.96
            and old_text == new_text
            and old_type == new_type
            and old_risk == new_risk
            and abs(old_confidence - new_confidence) < 0.001
            and old_review == new_review
        )

        if is_unchanged:
            change_type = "unchanged"
            summary["unchanged"] += 1
        else:
            change_type = "changed"
            summary["changed"] += 1

        clause_changes.append(
            {
                "change_type": change_type,
                "similarity": round(similarity, 4),
                "base_clause": build_clause_snapshot(base_clause),
                "comparison_clause": build_clause_snapshot(
                    comparison_clause
                ),
                "risk_change": risk_change,
                "confidence_change": round(
                    new_confidence - old_confidence,
                    4,
                ),
                "manual_review_change": manual_review_change,
            }
        )

    for index, comparison_clause in enumerate(comparison_clauses):
        if index in used_comparison_indexes:
            continue

        summary["added"] += 1

        clause_changes.append(
            {
                "change_type": "added",
                "similarity": 0.0,
                "base_clause": None,
                "comparison_clause": build_clause_snapshot(
                    comparison_clause
                ),
                "risk_change": "added",
                "confidence_change": None,
                "manual_review_change": "added"
                if comparison_clause.get("needs_manual_review")
                else "unchanged",
            }
        )

    base_analysis = base_contract.get("analysis", {})
    comparison_analysis = comparison_contract.get("analysis", {})

    if not isinstance(base_analysis, dict):
        base_analysis = {}

    if not isinstance(comparison_analysis, dict):
        comparison_analysis = {}

    return {
        "base_contract": {
            "contract_id": base_contract.get("contract_id"),
            "filename": base_contract.get("filename"),
            "contract_type": base_analysis.get("contract_type"),
            "overall_risk": base_analysis.get("overall_risk"),
            "total_clauses": len(base_clauses),
        },
        "comparison_contract": {
            "contract_id": comparison_contract.get("contract_id"),
            "filename": comparison_contract.get("filename"),
            "contract_type": comparison_analysis.get("contract_type"),
            "overall_risk": comparison_analysis.get("overall_risk"),
            "total_clauses": len(comparison_clauses),
        },
        "summary": summary,
        "clause_changes": clause_changes,
    }