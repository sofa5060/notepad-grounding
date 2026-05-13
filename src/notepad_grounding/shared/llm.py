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


@dataclass(frozen=True)
class CellChoice:
    cell_id: str
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


def resolve_openai_model(model: str | None = None) -> str:
    return model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
