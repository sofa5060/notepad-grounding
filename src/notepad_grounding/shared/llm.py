from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from notepad_grounding.shared.env import load_env_file
from notepad_grounding.shared.images import image_to_data_url

DEFAULT_OPENAI_MODEL = "gpt-5.4"


class VisionClient(Protocol):
    def choose_cell(self, *, query: str, image: Image.Image, cell_ids: list[str]) -> "CellChoice":
        """Choose the grid cell most likely to contain the visual target."""

    def locate_icon(self, *, query: str, image: Image.Image) -> "IconDetection":
        """Return a crop-local icon bounding box for the visual target."""


@dataclass(frozen=True)
class CellChoice:
    cell_id: str
    confidence: float
    rationale: str


@dataclass(frozen=True)
class IconDetection:
    target_visible: bool
    icon_bbox: tuple[int, int, int, int]
    confidence: float
    rationale: str


class OpenAIVisionClient:
    def __init__(self, *, model: str | None = None) -> None:
        load_env_file()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing OpenAI SDK. Run `uv sync` after pulling latest.") from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for the llm-visual flow. Add it to .env or set it in the shell.")
        self._client = OpenAI()
        self._model = resolve_openai_model(model)

    def choose_cell(self, *, query: str, image: Image.Image, cell_ids: list[str]) -> CellChoice:
        prompt = (
            "You are helping a Windows desktop visual grounding system. "
            f"Find the grid cell that most likely contains the desktop icon or shortcut for: {query!r}. "
            "Return JSON only with keys cell_id, confidence, rationale. "
            f"Valid cell_id values are: {', '.join(cell_ids)}. "
            "Do not return pixel coordinates."
        )
        response = self._client.responses.create(
            model=self._model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_to_data_url(image), "detail": "high"},
                    ],
                }
            ],
        )
        return parse_cell_choice(response.output_text, valid_cell_ids=cell_ids)

    def locate_icon(self, *, query: str, image: Image.Image) -> IconDetection:
        prompt = (
            "You are helping a Windows desktop visual grounding system. "
            f"Locate the icon graphic for the desktop shortcut or target: {query!r}. "
            "Return JSON only with keys target_visible, icon_bbox, confidence, rationale. "
            "icon_bbox must be crop-local pixel coordinates [x1, y1, x2, y2] around the icon graphic, not the label. "
            "Do not return full-screen coordinates."
        )
        response = self._client.responses.create(
            model=self._model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_to_data_url(image), "detail": "high"},
                    ],
                }
            ],
        )
        return parse_icon_detection(response.output_text, image_size=image.size)


def parse_cell_choice(text: str, *, valid_cell_ids: list[str]) -> CellChoice:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {text}") from exc

    cell_id = str(payload.get("cell_id", "")).strip()
    if cell_id not in valid_cell_ids:
        raise ValueError(f"LLM returned invalid cell_id {cell_id!r}; expected one of {valid_cell_ids}")

    confidence = float(payload.get("confidence", 0))
    confidence = max(0.0, min(1.0, confidence))
    rationale = str(payload.get("rationale", "")).strip()
    return CellChoice(cell_id=cell_id, confidence=confidence, rationale=rationale)


def parse_icon_detection(text: str, *, image_size: tuple[int, int]) -> IconDetection:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {text}") from exc

    target_visible = bool(payload.get("target_visible", False))
    raw_bbox = payload.get("icon_bbox")
    if not isinstance(raw_bbox, list | tuple) or len(raw_bbox) != 4:
        raise ValueError(f"LLM returned invalid icon_bbox: {raw_bbox!r}")

    bbox = tuple(round(float(value)) for value in raw_bbox)
    width, height = image_size
    clamped = (
        max(0, min(bbox[0], width)),
        max(0, min(bbox[1], height)),
        max(0, min(bbox[2], width)),
        max(0, min(bbox[3], height)),
    )
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        raise ValueError(f"LLM returned empty icon_bbox: {raw_bbox!r}")

    confidence = float(payload.get("confidence", 0))
    confidence = max(0.0, min(1.0, confidence))
    rationale = str(payload.get("rationale", "")).strip()
    return IconDetection(
        target_visible=target_visible,
        icon_bbox=clamped,
        confidence=confidence,
        rationale=rationale,
    )


def resolve_openai_model(model: str | None = None) -> str:
    return model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
