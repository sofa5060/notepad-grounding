from __future__ import annotations

import json
import os
from typing import Protocol

from PIL import Image

from notepad_grounding.env import load_env_file
from notepad_grounding.images import image_to_data_url
from notepad_grounding.models import CellChoice
from notepad_grounding.models import ClickGridChoice
from notepad_grounding.models import IconDetection
from notepad_grounding.prompts import build_click_grid_prompt

DEFAULT_OPENAI_MODEL = "gpt-5.4"


class VisionClient(Protocol):
    def choose_cell(self, *, query: str, image: Image.Image, cell_ids: list[str]) -> CellChoice:
        """Choose the grid cell most likely to contain the visual target."""

    def revise_cell_choice(
        self,
        *,
        query: str,
        image: Image.Image,
        cell_ids: list[str],
        rejected_cell_ids: list[str],
        reviewer_rationale: str,
        previous_response_id: str | None,
    ) -> CellChoice:
        """Choose a different grid cell after a reviewer rejected previous choices."""

    def choose_click_grid_cell(
        self,
        *,
        query: str,
        image: Image.Image,
        cell_ids: list[str],
        rejected_cell_ids: list[str] | None = None,
        previous_response_id: str | None = None,
    ) -> ClickGridChoice:
        """Choose the labeled row/column grid cell closest to the visual target."""


class OpenAIVisionClient:
    def __init__(self, *, model: str | None = None) -> None:
        load_env_file()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing OpenAI SDK. Run `uv sync` after pulling latest.") from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required. Add it to .env or set it in the shell.")
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

        choice, prev_response_id = self._try_parse_cell_choice(response, cell_ids, image)
        return CellChoice(
            cell_id=choice.cell_id,
            confidence=choice.confidence,
            rationale=choice.rationale,
            response_id=prev_response_id,
        )

    def revise_cell_choice(
        self,
        *,
        query: str,
        image: Image.Image,
        cell_ids: list[str],
        rejected_cell_ids: list[str],
        reviewer_rationale: str,
        previous_response_id: str | None,
    ) -> CellChoice:
        remaining = [cell_id for cell_id in cell_ids if cell_id not in set(rejected_cell_ids)]
        prompt = (
            "You are helping a Windows desktop visual grounding system. "
            f"You previously selected grid cell(s) {', '.join(rejected_cell_ids)} for target {query!r}, "
            "but a reviewer inspected the selected crop and rejected it because it did not contain the target.\n\n"
            f"Reviewer rationale: {reviewer_rationale}\n\n"
            f"Choose a different grid cell that most likely contains the desktop icon or shortcut for {query!r}. "
            f"Do NOT choose any rejected cell: {', '.join(rejected_cell_ids)}. "
            "Return JSON only with keys cell_id, confidence, rationale. "
            f"Valid cell_id values are: {', '.join(remaining)}. "
            "Do not return pixel coordinates."
        )
        request = {
            "model": self._model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_to_data_url(image), "detail": "high"},
                    ],
                }
            ],
        }
        if previous_response_id:
            request["previous_response_id"] = previous_response_id

        response = self._client.responses.create(**request)
        choice = parse_cell_choice(response.output_text, valid_cell_ids=remaining)
        return CellChoice(
            cell_id=choice.cell_id,
            confidence=choice.confidence,
            rationale=choice.rationale,
            response_id=response.id,
        )

    def choose_click_grid_cell(
        self,
        *,
        query: str,
        image: Image.Image,
        cell_ids: list[str],
        rejected_cell_ids: list[str] | None = None,
        previous_response_id: str | None = None,
    ) -> ClickGridChoice:
        rejected_cell_ids = rejected_cell_ids or []
        prompt = build_click_grid_prompt(query=query, cell_ids=cell_ids, rejected_cell_ids=rejected_cell_ids)
        request = {
            "model": self._model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_to_data_url(image), "detail": "high"},
                    ],
                }
            ],
        }
        if previous_response_id:
            request["previous_response_id"] = previous_response_id

        response = self._client.responses.create(**request)
        valid_cell_ids = [cell_id for cell_id in cell_ids if cell_id not in set(rejected_cell_ids)]
        choice = parse_click_grid_choice(response.output_text, valid_cell_ids=valid_cell_ids)
        return ClickGridChoice(
            cell_id=choice.cell_id,
            confidence=choice.confidence,
            rationale=choice.rationale,
            response_id=response.id,
        )

    def _try_parse_cell_choice(
        self,
        response,
        cell_ids: list[str],
        image: Image.Image,
        max_retries: int = 2,
    ) -> tuple[CellChoice, str]:
        prev_response_id = response.id
        for _ in range(max_retries + 1):
            try:
                choice = parse_cell_choice(response.output_text, valid_cell_ids=cell_ids)
                return choice, prev_response_id
            except ValueError as exc:
                invalid_cell_id = self._extract_invalid_cell_id(str(exc))
                correction_prompt = (
                    f"Your previous response returned an invalid cell_id '{invalid_cell_id}'. "
                    f"Valid cell_id values are: {', '.join(cell_ids)}. "
                    "Please return a JSON object with a valid cell_id."
                )
                response = self._client.responses.create(
                    model=self._model,
                    previous_response_id=prev_response_id,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": correction_prompt},
                                {"type": "input_image", "image_url": image_to_data_url(image), "detail": "high"},
                            ],
                        }
                    ],
                )
                prev_response_id = response.id
        raise ValueError(f"LLM failed to return a valid cell_id after {max_retries} retries")

    @staticmethod
    def _extract_invalid_cell_id(error_msg: str) -> str:
        start = error_msg.find("'") + 1
        end = error_msg.find("'", start)
        if start > 0 and end > start:
            return error_msg[start:end]
        return "unknown"


def parse_cell_choice(text: str, *, valid_cell_ids: list[str]) -> CellChoice:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {text}") from exc

    cell_id = str(payload.get("cell_id", "")).strip()
    if cell_id not in valid_cell_ids:
        raise ValueError(f"LLM returned invalid cell_id {cell_id!r}; expected one of {valid_cell_ids}")

    confidence = max(0.0, min(1.0, float(payload.get("confidence", 0))))
    rationale = str(payload.get("rationale", "")).strip()
    return CellChoice(cell_id=cell_id, confidence=confidence, rationale=rationale)


def parse_click_grid_choice(text: str, *, valid_cell_ids: list[str]) -> ClickGridChoice:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {text}") from exc

    cell_id = str(payload.get("cell_id", "")).strip()
    if cell_id not in valid_cell_ids:
        raise ValueError(f"LLM returned invalid cell_id {cell_id!r}; expected one of {valid_cell_ids}")

    confidence = max(0.0, min(1.0, float(payload.get("confidence", 0))))
    rationale = str(payload.get("rationale", "")).strip()
    return ClickGridChoice(cell_id=cell_id, confidence=confidence, rationale=rationale)


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

    confidence = max(0.0, min(1.0, float(payload.get("confidence", 0))))
    rationale = str(payload.get("rationale", "")).strip()
    return IconDetection(
        target_visible=target_visible,
        icon_bbox=clamped,
        confidence=confidence,
        rationale=rationale,
    )


def resolve_openai_model(model: str | None = None) -> str:
    return model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

