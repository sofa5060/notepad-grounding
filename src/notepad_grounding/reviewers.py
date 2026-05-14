from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from PIL import Image

from notepad_grounding.env import load_env_file
from notepad_grounding.images import draw_box_on_image
from notepad_grounding.images import image_to_data_url
from notepad_grounding.models import DesktopReviewResult
from notepad_grounding.models import IconDetection
from notepad_grounding.models import ReviewResultModel
from notepad_grounding.models import TargetReviewResult
from notepad_grounding.models import TargetReviewResultModel
from notepad_grounding.prompts import build_bbox_initial_prompt
from notepad_grounding.prompts import build_bbox_validation_prompt
from notepad_grounding.vision import DEFAULT_OPENAI_MODEL
from notepad_grounding.vision import parse_icon_detection


class TargetReviewer(Protocol):
    def review_target_crop(self, *, query: str, image: Image.Image) -> TargetReviewResult:
        """Return whether a selected crop contains the requested desktop target."""


class DesktopReviewer(Protocol):
    def review_desktop_state(
        self,
        *,
        action: str,
        expected: str,
        image: Image.Image,
    ) -> DesktopReviewResult:
        """Review the current desktop state after an automation action."""


class BboxReviewer(Protocol):
    def review_bbox(
        self,
        *,
        query: str,
        image: Image.Image,
        max_iterations: int = 3,
        debug_dir: Path | None = None,
    ) -> IconDetection:
        """Locate and review a crop-local icon bbox."""


class OpenAITargetReviewer:
    """Vision reviewer that validates a candidate grid crop before the search descends."""

    def __init__(self, *, model: str | None = None) -> None:
        load_env_file()
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing OpenAI/instructor SDK. Run `uv sync` after pulling latest.") from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required. Add it to .env or set it in the shell.")
        self._client = instructor.from_openai(OpenAI())
        self._model = resolve_openai_reviewer_model(model)

    def review_target_crop(self, *, query: str, image: Image.Image) -> TargetReviewResult:
        prompt = (
            "You are a strict reviewer for a Windows desktop visual grounding system.\n\n"
            f"Target query: {query!r}\n\n"
            "The image is a crop from a selected grid cell. Decide whether this crop contains "
            "the requested desktop item. Accept the crop if it contains either:\n"
            "1. recognizable visual evidence of the requested app/icon/shortcut, or\n"
            "2. visible label text matching the query.\n\n"
            "Reject the crop if the target is not visible, if it only contains a different app, "
            "or if the evidence is too ambiguous to continue safely."
        )
        parsed: TargetReviewResultModel = self._client.chat.completions.create(
            model=self._model,
            response_model=TargetReviewResultModel,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_url(image), "detail": "high"},
                        },
                    ],
                }
            ],
        )

        return TargetReviewResult(
            contains_target=parsed.contains_target,
            confidence=parsed.confidence,
            rationale=parsed.rationale.strip(),
            visible_evidence=parsed.visible_evidence.strip(),
        )


class OpenAIDesktopReviewer:
    """LLM reviewer for desktop automation state."""

    def __init__(self, *, model: str | None = None) -> None:
        load_env_file()
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing OpenAI/instructor SDK. Run `uv sync` after pulling latest.") from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required. Add it to .env or set it in the shell.")
        self._client = instructor.from_openai(OpenAI())
        self._model = resolve_openai_reviewer_model(model)

    def review_desktop_state(
        self,
        *,
        action: str,
        expected: str,
        image: Image.Image,
    ) -> DesktopReviewResult:
        prompt = (
            "You are a desktop automation reviewer. You validate whether an action succeeded "
            "by looking at the current screenshot.\n\n"
            f"Action just performed: {action}\n"
            f"Expected state: {expected}\n\n"
            "Look at the screenshot and determine:\n"
            "1. Is the expected state achieved?\n"
            "2. Is there an unexpected pop-up, dialog, or wrong window open?\n"
            "3. If something is wrong, what is the exact recovery action?"
        )

        parsed: ReviewResultModel = self._client.chat.completions.create(
            model=self._model,
            response_model=ReviewResultModel,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_url(image), "detail": "high"},
                        },
                    ],
                }
            ],
        )

        return DesktopReviewResult(
            status=parsed.status.lower().strip(),
            action_needed=parsed.action_needed.strip(),
            rationale=parsed.rationale.strip(),
        )


class OpenAIBboxReviewer:
    """Iterative bbox reviewer used as the fallback final precision method."""

    def __init__(self, *, model: str | None = None) -> None:
        load_env_file()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing OpenAI SDK. Run `uv sync` after pulling latest.") from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required. Add it to .env or set it in the shell.")
        self._client = OpenAI()
        self._model = resolve_openai_reviewer_model(model)

    def review_bbox(
        self,
        *,
        query: str,
        image: Image.Image,
        max_iterations: int = 3,
        debug_dir: Path | None = None,
    ) -> IconDetection:
        response = self._client.responses.create(
            model=self._model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": build_bbox_initial_prompt(query=query)},
                        {"type": "input_image", "image_url": image_to_data_url(image), "detail": "high"},
                    ],
                }
            ],
        )
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
            response = self._client.responses.create(
                model=self._model,
                previous_response_id=request_previous_response_id,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": build_bbox_validation_prompt()},
                            {"type": "input_image", "image_url": image_to_data_url(annotated), "detail": "high"},
                        ],
                    }
                ],
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
            clamped = (
                max(0, min(bbox[0], width)),
                max(0, min(bbox[1], height)),
                max(0, min(bbox[2], width)),
                max(0, min(bbox[3], height)),
            )
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


def _write_bbox_debug_json(debug_dir: Path | None, filename: str, payload: dict) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_openai_reviewer_model(model: str | None = None) -> str:
    return (
        model
        or os.environ.get("OPENAI_REVIEWER_MODEL")
        or os.environ.get("OPENAI_JUDGE_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or DEFAULT_OPENAI_MODEL
    )
