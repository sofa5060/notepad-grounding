from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import Field

if TYPE_CHECKING:
    from notepad_grounding_paper.images import Box
    from notepad_grounding_paper.llm import OpenAIReviewer
    from notepad_grounding_paper.llm import OpenAIVisionClient


@dataclass(frozen=True)
class CellChoice:
    cell_id: str
    confidence: float
    rationale: str
    response_id: str | None = None


@dataclass(frozen=True)
class IconDetection:
    target_visible: bool
    icon_bbox: tuple[int, int, int, int]
    confidence: float
    rationale: str


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
class TargetReviewAttempt:
    attempt_index: int
    selected_cell_id: str
    selected_box: Box
    reviewed_crop_box: Box
    crop_image: str
    contains_target: bool
    confidence: float
    rationale: str
    visible_evidence: str


@dataclass(frozen=True)
class VisualSearchStep:
    round_index: int
    crop_box: Box
    selected_cell_id: str
    selected_box: Box
    confidence: float
    rationale: str
    grid_image: str
    review_attempts: list[TargetReviewAttempt]


@dataclass(frozen=True)
class FinalDetectionStep:
    crop_box: Box
    icon_bbox_local: Box
    icon_bbox_screen: Box
    confidence: float
    rationale: str
    crop_image: str
    detection_image: str


@dataclass(frozen=True)
class FinalClickPointStep:
    crop_box: Box
    coarse_point_id: str
    fine_point_id: str
    crop_point: tuple[int, int]
    screen_point: tuple[int, int]
    first_overlay_image: str
    first_result_json: str
    refinement_crop_box: Box
    refinement_crop_image: str
    second_overlay_image: str
    second_result_json: str
    final_image: str
    full_image: str
    result_json: str
    confidence: float
    rationale: str


@dataclass(frozen=True)
class VisualSearchResult:
    query: str
    center: tuple[int, int]
    final_box: Box
    steps: list[VisualSearchStep]
    final_method: str
    final_detection: FinalDetectionStep | None
    final_click_point: FinalClickPointStep | None
    output_dir: str
    result_json: str
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
    result_json: str


@dataclass(frozen=True)
class PostResult:
    post_id: int
    status: str
    filename: str
    center: tuple[int, int] | None
    error: str | None


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


class TargetReviewResultModel(BaseModel):
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
