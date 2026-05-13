from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class CellChoiceModel(BaseModel):
    """Structured output for single grid cell selection."""

    cell_id: str = Field(..., description="The ID of the selected grid cell")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1")
    rationale: str = Field(..., description="Why this cell was chosen")


class CellsChoiceModel(BaseModel):
    """Structured output for multiple grid cell selection."""

    cell_ids: list[str] = Field(..., description="All cell IDs containing the icon")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1")
    rationale: str = Field(..., description="Why these cells were chosen")


class ReviewResultModel(BaseModel):
    """Structured output for state review/validation."""

    status: str = Field(
        ...,
        description="One of: success, wrong_app, pop_up, error, retry",
    )
    action_needed: str = Field(
        ...,
        description="Recovery action like 'proceed', 'close window and retry', 'click Replace button'",
    )
    rationale: str = Field(..., description="Explanation of the current state")
