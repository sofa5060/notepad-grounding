from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from notepad_grounding.shared.images import image_to_data_url
from notepad_grounding.shared.llm import resolve_openai_model


class ReviewClient(Protocol):
    def review_state(
        self,
        *,
        action: str,
        expected: str,
        image: Image.Image,
    ) -> "ReviewResult":
        """Review the current screen state after an action was performed."""


@dataclass(frozen=True)
class ReviewResult:
    status: str  # "success", "wrong_app", "pop_up", "error", "retry"
    action_needed: str  # what recovery action to take
    rationale: str


class OpenAIReviewClient:
    def __init__(self, *, model: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing OpenAI SDK. Run `uv sync`.") from exc

        self._client = OpenAI()
        self._model = resolve_openai_model(model)

    def review_state(
        self,
        *,
        action: str,
        expected: str,
        image: Image.Image,
    ) -> ReviewResult:
        prompt = (
            "You are a desktop automation reviewer. You validate whether an action succeeded "
            "by looking at the current screenshot.\n\n"
            f"Action just performed: {action}\n"
            f"Expected state: {expected}\n\n"
            "Look at the screenshot and determine:\n"
            "1. Is the expected state achieved?\n"
            "2. Is there an unexpected pop-up, dialog, or wrong window open?\n"
            "3. If something is wrong, what is the exact recovery action?\n\n"
            "Return JSON only with keys: status, action_needed, rationale.\n"
            "status must be one of: success, wrong_app, pop_up, error, retry.\n"
            "action_needed: a brief instruction like 'click Replace button', "
            "'close wrong window and retry', 'proceed', 'wait longer', etc."
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
        return parse_review_result(response.output_text)


def parse_review_result(text: str) -> ReviewResult:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {text}") from exc

    status = str(payload.get("status", "error")).strip().lower()
    valid_statuses = {"success", "wrong_app", "pop_up", "error", "retry"}
    if status not in valid_statuses:
        status = "error"

    action_needed = str(payload.get("action_needed", "retry")).strip()
    rationale = str(payload.get("rationale", "")).strip()

    return ReviewResult(
        status=status,
        action_needed=action_needed,
        rationale=rationale,
    )
