from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from notepad_grounding.shared.env import load_env_file
from notepad_grounding.shared.images import image_to_data_url
from notepad_grounding.shared.llm import DEFAULT_OPENAI_MODEL
from notepad_grounding.shared.schemas import GridJudgeResultModel


class GridJudgeClient(Protocol):
    def judge_crop(self, *, query: str, image: Image.Image) -> "GridJudgeResult":
        """Return whether a selected crop contains the requested desktop target."""


@dataclass(frozen=True)
class GridJudgeResult:
    contains_target: bool
    confidence: float
    rationale: str
    visible_evidence: str = ""


class OpenAIGridJudgeClient:
    """Vision judge that validates a candidate grid crop before the search descends."""

    def __init__(self, *, model: str | None = None) -> None:
        load_env_file()
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing OpenAI/instructor SDK. Run `uv sync` after pulling latest.") from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for the grid judge. Add it to .env or set it in the shell.")
        self._client = instructor.from_openai(OpenAI())
        self._model = resolve_openai_judge_model(model)

    def judge_crop(self, *, query: str, image: Image.Image) -> GridJudgeResult:
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
        parsed: GridJudgeResultModel = self._client.chat.completions.create(
            model=self._model,
            response_model=GridJudgeResultModel,
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

        return GridJudgeResult(
            contains_target=parsed.contains_target,
            confidence=parsed.confidence,
            rationale=parsed.rationale.strip(),
            visible_evidence=parsed.visible_evidence.strip(),
        )


def resolve_openai_judge_model(model: str | None = None) -> str:
    return model or os.environ.get("OPENAI_JUDGE_MODEL") or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
