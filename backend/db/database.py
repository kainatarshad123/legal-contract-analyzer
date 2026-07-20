"""SQLite persistence helpers for contracts and RAG chunks."""

import json
import os
import sqlite3
from datetime import datetime
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "contracts.db")


def get_db_connection() -> sqlite3.Connection:
    """Create a SQLite connection with dictionary-like rows."""

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create all required database tables."""

    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS contracts (
                contract_id TEXT PRIMARY KEY,
                filename TEXT,
                content_type TEXT,
                contract_text TEXT,
                clauses_json TEXT,
                analysis_json TEXT,
                created_at TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS contract_chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                clause_number INTEGER,
                clause_type TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(contract_id, chunk_index),
                FOREIGN KEY(contract_id)
                    REFERENCES contracts(contract_id)
                    ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_contract_chunks_contract_id
            ON contract_chunks(contract_id)
            """
        )


def save_contract_to_db(
    contract_id: str,
    filename: str | None,
    content_type: str | None,
    contract_text: str,
    clauses: list[dict[str, Any]],
    analysis: dict[str, Any],
) -> None:
    """Insert or replace one analyzed contract."""

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO contracts (
                contract_id,
                filename,
                content_type,
                contract_text,
                clauses_json,
                analysis_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                contract_id,
                filename,
                content_type,
                contract_text,
                json.dumps(clauses),
                json.dumps(analysis),
                datetime.now().isoformat(),
            ),
        )


def save_contract_chunks(
    contract_id: str,
    chunks: list[dict[str, Any]],
) -> int:
    """Replace all stored RAG chunks for one contract.

    Returns the number of chunks written.
    """

    created_at = datetime.now().isoformat()

    with get_db_connection() as conn:
        contract_exists = conn.execute(
            """
            SELECT 1
            FROM contracts
            WHERE contract_id = ?
            """,
            (contract_id,),
        ).fetchone()

        if contract_exists is None:
            raise ValueError(
                f"Cannot save chunks: contract {contract_id!r} does not exist."
            )

        conn.execute(
            """
            DELETE FROM contract_chunks
            WHERE contract_id = ?
            """,
            (contract_id,),
        )

        rows: list[tuple[Any, ...]] = []

        for fallback_index, chunk in enumerate(chunks):
            chunk_text = str(
                chunk.get("text")
                or chunk.get("chunk_text")
                or ""
            ).strip()

            if not chunk_text:
                continue

            raw_chunk_index = chunk.get("chunk_index", fallback_index)

            try:
                chunk_index = int(raw_chunk_index)
            except (TypeError, ValueError):
                chunk_index = fallback_index

            raw_clause_number = chunk.get("clause_number")

            try:
                clause_number = (
                    int(raw_clause_number)
                    if raw_clause_number is not None
                    else None
                )
            except (TypeError, ValueError):
                clause_number = None

            clause_type_value = chunk.get("clause_type")
            clause_type = (
                str(clause_type_value).strip()
                if clause_type_value is not None
                else None
            )

            rows.append(
                (
                    contract_id,
                    chunk_index,
                    chunk_text,
                    clause_number,
                    clause_type,
                    created_at,
                )
            )

        if rows:
            conn.executemany(
                """
                INSERT INTO contract_chunks (
                    contract_id,
                    chunk_index,
                    chunk_text,
                    clause_number,
                    clause_type,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    return len(rows)


def load_contract_chunks(
    contract_id: str,
) -> list[dict[str, Any]]:
    """Load stored RAG chunks for one contract."""

    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                chunk_id,
                contract_id,
                chunk_index,
                chunk_text,
                clause_number,
                clause_type,
                created_at
            FROM contract_chunks
            WHERE contract_id = ?
            ORDER BY chunk_index ASC
            """,
            (contract_id,),
        ).fetchall()

    return [
        {
            "chunk_id": row["chunk_id"],
            "contract_id": row["contract_id"],
            "chunk_index": row["chunk_index"],
            "text": row["chunk_text"],
            "clause_number": row["clause_number"],
            "clause_type": row["clause_type"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def load_contract_from_db(contract_id: str) -> dict[str, Any] | None:
    """Load one contract and deserialize its JSON fields."""

    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM contracts
            WHERE contract_id = ?
            """,
            (contract_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "contract_id": row["contract_id"],
        "filename": row["filename"],
        "content_type": row["content_type"],
        "text": row["contract_text"],
        "clauses": json.loads(row["clauses_json"] or "[]"),
        "analysis": json.loads(row["analysis_json"] or "{}"),
        "created_at": row["created_at"],
    }


def delete_contract_from_db(contract_id: str) -> bool:
    """Delete a contract and all of its stored chunks."""

    with get_db_connection() as conn:
        # Explicit deletion keeps this safe for databases created before
        # foreign-key cascading was enabled.
        conn.execute(
            """
            DELETE FROM contract_chunks
            WHERE contract_id = ?
            """,
            (contract_id,),
        )

        cursor = conn.execute(
            """
            DELETE FROM contracts
            WHERE contract_id = ?
            """,
            (contract_id,),
        )

        return cursor.rowcount > 0


def list_contracts_from_db() -> list[dict[str, Any]]:
    """Return saved-contract metadata ordered newest first."""

    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT contract_id, filename, content_type, created_at
            FROM contracts
            ORDER BY created_at DESC
            """
        ).fetchall()

    return [
        {
            "contract_id": row["contract_id"],
            "filename": row["filename"],
            "content_type": row["content_type"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
