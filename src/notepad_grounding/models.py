from __future__ import annotations

from pydantic import BaseModel, Field


class IconBBox(BaseModel):
    left: int = Field(description="Left edge of the icon bounding box in pixels")
    top: int = Field(description="Top edge of the icon bounding box in pixels")
    right: int = Field(description="Right edge of the icon bounding box in pixels")
    bottom: int = Field(description="Bottom edge of the icon bounding box in pixels")


class IconLocation(BaseModel):
    x: int = Field(description="X coordinate of the icon center in pixels")
    y: int = Field(description="Y coordinate of the icon center in pixels")
    bbox: IconBBox = Field(description="Bounding box of the visible icon graphic area")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    rationale: str = Field(description="Short explanation of the location choice")


class AppOpenReview(BaseModel):
    opened_expected_app: bool


class NotepadOpenTimeoutError(RuntimeError):
    pass


class AppOpenVerificationError(RuntimeError):
    pass


class AppCloseTimeoutError(RuntimeError):
    pass
