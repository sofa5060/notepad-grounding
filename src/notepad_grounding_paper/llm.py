from __future__ import annotations

import base64
import json
import os
from dataclasses import replace
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from notepad_grounding_paper.models import CellChoice
from notepad_grounding_paper.models import DesktopReviewResult
from notepad_grounding_paper.models import ReviewResultModel
from notepad_grounding_paper.models import TargetReviewResult
from notepad_grounding_paper.models import TargetReviewResultModel
from notepad_grounding_paper.prompts import build_cell_choice_prompt
from notepad_grounding_paper.prompts import build_choice_correction_prompt
from notepad_grounding_paper.prompts import build_click_grid_prompt
from notepad_grounding_paper.prompts import build_desktop_review_prompt
from notepad_grounding_paper.prompts import build_revise_cell_choice_prompt
from notepad_grounding_paper.prompts import build_target_grid_review_prompt
from notepad_grounding_paper.prompts import build_target_review_prompt

DEFAULT_OPENAI_MODEL = "gpt-5.4"


def resolve_openai_model(model: str | None = None) -> str:
    return model or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL


def _create_client() -> OpenAI:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Add it to .env or set it in the shell.")
    return OpenAI()


def _image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _user_message(prompt: str, image: Image.Image) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": _image_to_data_url(image), "detail": "high"},
            ],
        }
    ]


class _OpenAIClient:
    def __init__(self, *, model: str | None = None) -> None:
        self._client = _create_client()
        self._model = resolve_openai_model(model)

    def _ask_model(self, *, prompt: str, image: Image.Image, previous_response_id: str | None = None):
        request = {"model": self._model, "input": _user_message(prompt, image)}
        if previous_response_id:
            request["previous_response_id"] = previous_response_id
        return self._client.responses.create(**request)


class OpenAIVisionClient(_OpenAIClient):
    def choose_cell(self, *, query: str, image: Image.Image, cell_ids: list[str]) -> CellChoice:
        return self._choose_cell_id(
            prompt=build_cell_choice_prompt(query=query, cell_ids=cell_ids), image=image, valid_cell_ids=cell_ids
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
        rejected = set(rejected_cell_ids)
        remaining = [cell_id for cell_id in cell_ids if cell_id not in rejected]
        return self._choose_cell_id(
            prompt=build_revise_cell_choice_prompt(
                query=query,
                rejected_cell_ids=rejected_cell_ids,
                reviewer_rationale=reviewer_rationale,
                valid_cell_ids=remaining,
            ),
            image=image,
            valid_cell_ids=remaining,
            previous_response_id=previous_response_id,
        )

    def choose_click_grid_cell(
        self,
        *,
        query: str,
        image: Image.Image,
        cell_ids: list[str],
        rejected_cell_ids: list[str] | None = None,
        previous_response_id: str | None = None,
    ) -> CellChoice:
        rejected_cell_ids = rejected_cell_ids or []
        rejected = set(rejected_cell_ids)
        valid_cell_ids = [cell_id for cell_id in cell_ids if cell_id not in rejected]
        return self._choose_cell_id(
            prompt=build_click_grid_prompt(
                query=query, valid_cell_ids=valid_cell_ids, rejected_cell_ids=rejected_cell_ids
            ),
            image=image,
            valid_cell_ids=valid_cell_ids,
            previous_response_id=previous_response_id,
        )

    def _choose_cell_id(
        self,
        *,
        prompt: str,
        image: Image.Image,
        valid_cell_ids: list[str],
        previous_response_id: str | None = None,
        max_retries: int = 2,
    ) -> CellChoice:
        last_error: ValueError | None = None
        for _ in range(max_retries + 1):
            response = self._ask_model(prompt=prompt, image=image, previous_response_id=previous_response_id)
            try:
                choice = parse_cell_choice(response.output_text, valid_cell_ids=valid_cell_ids)
                return replace(choice, response_id=response.id)
            except ValueError as exc:
                last_error = exc
                prompt = build_choice_correction_prompt(error=str(exc), valid_cell_ids=valid_cell_ids)
                previous_response_id = response.id
        raise ValueError(f"LLM failed to return a valid cell_id after {max_retries} retries") from last_error


class OpenAIReviewer(_OpenAIClient):
    """OpenAI reviewer for target crops and desktop state."""

    def review_target_crop(self, *, query: str, image: Image.Image) -> TargetReviewResult:
        return self._review_target(prompt=build_target_review_prompt(query=query), image=image)

    def review_target_grid_cell(self, *, query: str, image: Image.Image) -> TargetReviewResult:
        return self._review_target(prompt=build_target_grid_review_prompt(query=query), image=image)

    def _review_target(self, *, prompt: str, image: Image.Image) -> TargetReviewResult:
        parsed: TargetReviewResultModel = self._ask_structured_reviewer(
            prompt=prompt, image=image, response_model=TargetReviewResultModel
        )

        return TargetReviewResult(
            contains_target=parsed.contains_target,
            confidence=parsed.confidence,
            rationale=parsed.rationale.strip(),
            visible_evidence=parsed.visible_evidence.strip(),
        )

    def review_desktop_state(self, *, action: str, expected: str, image: Image.Image) -> DesktopReviewResult:
        parsed: ReviewResultModel = self._ask_structured_reviewer(
            prompt=build_desktop_review_prompt(action=action, expected=expected),
            image=image,
            response_model=ReviewResultModel,
        )

        return DesktopReviewResult(
            status=parsed.status.lower().strip(),
            action_needed=parsed.action_needed.strip(),
            rationale=parsed.rationale.strip(),
        )

    def _ask_structured_reviewer(self, *, prompt: str, image: Image.Image, response_model):
        response = self._client.responses.parse(
            model=self._model, input=_user_message(prompt, image), text_format=response_model
        )
        return response.output_parsed


def parse_cell_choice(text: str, *, valid_cell_ids: list[str]) -> CellChoice:
    payload = _load_json(text)
    cell_id = str(payload.get("cell_id", "")).strip()
    if cell_id not in valid_cell_ids:
        raise ValueError(f"LLM returned invalid cell_id {cell_id!r}; expected one of {valid_cell_ids}")
    confidence = max(0.0, min(1.0, float(payload.get("confidence", 0))))
    rationale = str(payload.get("rationale", "")).strip()
    return CellChoice(cell_id=cell_id, confidence=confidence, rationale=rationale)


def _load_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {text}") from exc
