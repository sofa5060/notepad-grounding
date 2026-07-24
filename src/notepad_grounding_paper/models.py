from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class CellChoice(BaseModel):
    cell_id: str = Field(description="The chosen grid cell id, e.g. R2C3")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence from 0.0 to 1.0")
    rationale: str = Field(description="Short explanation of the choice")


class TargetReview(BaseModel):
    contains_target: bool = Field(description="True when the highlighted cell contains the requested icon/app visual or a matching label")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence from 0.0 to 1.0")
    rationale: str = Field(description="Why the choice should be accepted or rejected")


class DesktopReview(BaseModel):
    status: str = Field(description="One of: success, wrong_app, pop_up, error, retry")
    action_needed: str = Field(description="Recovery action like 'proceed', 'close window and retry', 'click Replace button'")
    rationale: str = Field(description="Explanation of the current state")
