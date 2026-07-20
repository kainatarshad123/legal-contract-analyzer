"""Pydantic request schemas for contract Q&A and clause explanations."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for asking a question about a saved contract."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question about the uploaded contract.",
    )

    contract_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique ID of the saved contract.",
    )


class ExplainClauseRequest(BaseModel):
    """Request body for explaining one clause in plain English."""

    contract_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique ID of the saved contract.",
    )

    clause_number: int = Field(
        ...,
        ge=1,
        description="One-based number of the clause to explain.",
    )