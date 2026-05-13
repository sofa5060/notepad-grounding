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


class GridJudgeResultModel(BaseModel):
    """Structured output for validating a selected grid crop."""

    contains_target: bool = Field(
        ...,
        description="True when the crop contains the requested icon/app visual or a matching label.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1")
    rationale: str = Field(..., description="Why the crop should be accepted or rejected")
    visible_evidence: str = Field(
        "",
        description="Short description of visible evidence, such as icon appearance or label text.",
    )
