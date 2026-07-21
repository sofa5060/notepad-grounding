from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from notepad_grounding_paper.images import clamp_box
from notepad_grounding_paper.images import draw_box_on_image
from notepad_grounding_paper.images import image_to_data_url
from notepad_grounding_paper.models import CellChoice
from notepad_grounding_paper.models import DesktopReviewResult
from notepad_grounding_paper.models import IconDetection
from notepad_grounding_paper.models import ReviewResultModel
from notepad_grounding_paper.models import TargetReviewResult
from notepad_grounding_paper.models import TargetReviewResultModel
from notepad_grounding_paper.prompts import build_bbox_initial_prompt
from notepad_grounding_paper.prompts import build_bbox_validation_prompt
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


def _user_message(prompt: str, image: Image.Image) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": image_to_data_url(image), "detail": "high"},
            ],
        }
    ]


class _OpenAIClient:
    def __init__(self, *, model: str | None = None) -> None:
        self._client = _create_client()
        self._model = resolve_openai_model(model)

    def _ask_model(self, *, prompt: str, image: Image.Image, previous_response_id: str | None = None):
        request = {
            "model": self._model,
            "input": _user_message(prompt, image),
        }
        if previous_response_id:
            request["previous_response_id"] = previous_response_id
        return self._client.responses.create(**request)


class OpenAIVisionClient(_OpenAIClient):
    def choose_cell(self, *, query: str, image: Image.Image, cell_ids: list[str]) -> CellChoice:
        return self._choose_cell_id(
            prompt=build_cell_choice_prompt(query=query, cell_ids=cell_ids),
            image=image,
            valid_cell_ids=cell_ids,
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
        valid_cell_ids = [cell_id for cell_id in cell_ids if cell_id not in set(rejected_cell_ids)]
        return self._choose_cell_id(
            prompt=build_click_grid_prompt(query=query, cell_ids=cell_ids, rejected_cell_ids=rejected_cell_ids),
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
        response = self._ask_model(prompt=prompt, image=image, previous_response_id=previous_response_id)
        for attempt in range(max_retries + 1):
            try:
                choice = parse_cell_choice(response.output_text, valid_cell_ids=valid_cell_ids)
                return CellChoice(
                    cell_id=choice.cell_id,
                    confidence=choice.confidence,
                    rationale=choice.rationale,
                    response_id=response.id,
                )
            except ValueError as exc:
                if attempt == max_retries:
                    raise ValueError(f"LLM failed to return a valid cell_id after {max_retries} retries") from exc
                response = self._ask_model(
                    prompt=build_choice_correction_prompt(error=str(exc), valid_cell_ids=valid_cell_ids),
                    image=image,
                    previous_response_id=response.id,
                )


class OpenAIReviewer(_OpenAIClient):
    """OpenAI reviewer for target crops, desktop state, and bbox fallback."""

    def review_target_crop(self, *, query: str, image: Image.Image) -> TargetReviewResult:
        return self._review_target(prompt=build_target_review_prompt(query=query), image=image)

    def review_target_grid_cell(self, *, query: str, image: Image.Image) -> TargetReviewResult:
        return self._review_target(prompt=build_target_grid_review_prompt(query=query), image=image)

    def _review_target(self, *, prompt: str, image: Image.Image) -> TargetReviewResult:
        parsed: TargetReviewResultModel = self._ask_structured_reviewer(
            prompt=prompt,
            image=image,
            response_model=TargetReviewResultModel,
        )

        return TargetReviewResult(
            contains_target=parsed.contains_target,
            confidence=parsed.confidence,
            rationale=parsed.rationale.strip(),
            visible_evidence=parsed.visible_evidence.strip(),
        )

    def review_desktop_state(
        self,
        *,
        action: str,
        expected: str,
        image: Image.Image,
    ) -> DesktopReviewResult:
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

    def review_bbox(
        self,
        *,
        query: str,
        image: Image.Image,
        max_iterations: int = 3,
        debug_dir: Path | None = None,
    ) -> IconDetection:
        response = self._ask_model(prompt=build_bbox_initial_prompt(query=query), image=image)
        detection = parse_icon_detection(response.output_text, image_size=image.size)
        _write_bbox_debug_json(
            debug_dir,
            "bbox-initial-result.json",
            {
                "stage": "initial_detection",
                "response_id": response.id,
                "raw_output": response.output_text,
                "parsed_detection": asdict(detection),
            },
        )
        previous_response_id = response.id

        for iteration in range(1, max_iterations + 1):
            annotated = draw_box_on_image(
                image,
                detection.icon_bbox,
                label=f"bbox_v{iteration}",
                color=(255, 0, 0),
            )
            if debug_dir is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                annotated.save(debug_dir / f"bbox-review-{iteration:02d}.png")

            request_previous_response_id = previous_response_id
            response = self._ask_model(
                prompt=build_bbox_validation_prompt(),
                image=annotated,
                previous_response_id=request_previous_response_id,
            )
            previous_response_id = response.id

            debug_payload = {
                "stage": "bbox_review",
                "iteration": iteration,
                "response_id": response.id,
                "previous_response_id": request_previous_response_id,
                "input_bbox": detection.icon_bbox,
                "raw_output": response.output_text,
            }
            try:
                payload = json.loads(response.output_text)
            except json.JSONDecodeError:
                debug_payload["parse_error"] = "invalid_json"
                _write_bbox_debug_json(debug_dir, f"bbox-review-{iteration:02d}-result.json", debug_payload)
                break

            debug_payload["parsed_output"] = payload
            _write_bbox_debug_json(debug_dir, f"bbox-review-{iteration:02d}-result.json", debug_payload)

            confirmed = bool(payload.get("confirmed", False))
            raw_bbox = payload.get("corrected_icon_bbox")
            if confirmed or not isinstance(raw_bbox, list | tuple) or len(raw_bbox) != 4:
                break

            bbox = tuple(round(float(v)) for v in raw_bbox)
            width, height = image.size
            clamped = clamp_box(bbox, (0, 0, width, height))
            if clamped[2] > clamped[0] and clamped[3] > clamped[1]:
                detection = IconDetection(
                    target_visible=detection.target_visible,
                    icon_bbox=clamped,
                    confidence=float(payload.get("confidence", detection.confidence)),
                    rationale=str(payload.get("rationale", detection.rationale)).strip(),
                )

        _write_bbox_debug_json(
            debug_dir,
            "bbox-final-result.json",
            {
                "stage": "final_detection",
                "parsed_detection": asdict(detection),
            },
        )
        return detection

    def _ask_structured_reviewer(self, *, prompt: str, image: Image.Image, response_model):
        response = self._client.responses.parse(
            model=self._model,
            input=_user_message(prompt, image),
            text_format=response_model,
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


def parse_icon_detection(text: str, *, image_size: tuple[int, int]) -> IconDetection:
    payload = _load_json(text)
    raw_bbox = payload.get("icon_bbox")
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError(f"LLM returned invalid icon_bbox: {raw_bbox!r}")

    bbox = tuple(round(float(value)) for value in raw_bbox)
    width, height = image_size
    clamped = clamp_box(bbox, (0, 0, width, height))
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        raise ValueError(f"LLM returned empty icon_bbox: {raw_bbox!r}")

    confidence = max(0.0, min(1.0, float(payload.get("confidence", 0))))
    rationale = str(payload.get("rationale", "")).strip()
    return IconDetection(
        target_visible=bool(payload.get("target_visible", False)),
        icon_bbox=clamped,
        confidence=confidence,
        rationale=rationale,
    )


def _write_bbox_debug_json(debug_dir: Path | None, filename: str, payload: dict) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {text}") from exc
