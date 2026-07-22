from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import Field

if TYPE_CHECKING:
    from notepad_grounding_paper.llm import OpenAIReviewer
    from notepad_grounding_paper.llm import OpenAIVisionClient


@dataclass(frozen=True)
class CellChoice:
    cell_id: str
    confidence: float
    rationale: str
    response_id: str | None = None


@dataclass(frozen=True)
class TargetReviewResult:
    contains_target: bool
    confidence: float
    rationale: str
    visible_evidence: str = ""


@dataclass(frozen=True)
class DesktopReviewResult:
    status: str
    action_needed: str
    rationale: str


@dataclass(frozen=True)
class VisualSearchResult:
    query: str
    center: tuple[int, int]
    output_dir: str
    elapsed_seconds: float


@dataclass(frozen=True)
class FlowDependencies:
    client: OpenAIVisionClient
    reviewer: OpenAIReviewer


@dataclass(frozen=True)
class AutomationResult:
    query: str
    total_posts: int
    succeeded: int
    failed: int
    output_dir: str


class ReviewResultModel(BaseModel):
    """Structured output for state review/validation."""

    status: str = Field(..., description="One of: success, wrong_app, pop_up, error, retry")
    action_needed: str = Field(
        ..., description="Recovery action like 'proceed', 'close window and retry', 'click Replace button'"
    )
    rationale: str = Field(..., description="Explanation of the current state")


class TargetReviewResultModel(BaseModel):
    """Structured output for validating a selected grid crop."""

    contains_target: bool = Field(
        ..., description="True when the crop contains the requested icon/app visual or a matching label."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1")
    rationale: str = Field(..., description="Why the crop should be accepted or rejected")
    visible_evidence: str = Field(
        "", description="Short description of visible evidence, such as icon appearance or label text."
    )
