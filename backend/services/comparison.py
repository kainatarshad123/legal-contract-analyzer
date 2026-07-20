"""Routes for comparing saved contracts."""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from db.database import load_contract_from_db
from services.contract_comparison import compare_contracts


router = APIRouter(
    tags=["contract-comparison"],
)


class CompareContractsRequest(BaseModel):
    base_contract_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )
    comparison_contract_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )


def load_comparison_contract(
    contract_id: str,
    label: str,
) -> dict[str, Any]:
    contract_data = load_contract_from_db(contract_id)

    if contract_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{label} contract was not found.",
        )

    return contract_data


@router.post("/compare-contracts")
def compare_saved_contracts(
    request: CompareContractsRequest,
) -> dict[str, Any]:
    if request.base_contract_id == request.comparison_contract_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Select two different contracts to compare.",
        )

    base_contract = load_comparison_contract(
        request.base_contract_id,
        "Base",
    )

    comparison_contract = load_comparison_contract(
        request.comparison_contract_id,
        "Comparison",
    )

    return compare_contracts(
        base_contract,
        comparison_contract,
    )