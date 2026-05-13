from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from notepad_grounding.shared.images import image_to_data_url
from notepad_grounding.shared.llm import resolve_openai_model
from notepad_grounding.shared.schemas import ReviewResultModel


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
    """LLM reviewer that uses OpenAI structured outputs (Pydantic) for guaranteed format."""

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
            "3. If something is wrong, what is the exact recovery action?"
        )

        # Use beta.chat.completions.parse for guaranteed structured output via Pydantic
        completion = self._client.beta.chat.completions.parse(
            model=self._model,
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
            response_format=ReviewResultModel,
        )

        parsed: ReviewResultModel = completion.choices[0].message.parsed
        return ReviewResult(
            status=parsed.status.lower().strip(),
            action_needed=parsed.action_needed.strip(),
            rationale=parsed.rationale.strip(),
        )
