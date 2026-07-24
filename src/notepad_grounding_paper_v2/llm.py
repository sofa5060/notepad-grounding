from __future__ import annotations

import base64
import logging
import os
from io import BytesIO

import mss
from openai import OpenAI
from PIL import Image

from notepad_grounding_paper_v2.models import CellChoice
from notepad_grounding_paper_v2.models import DesktopReview
from notepad_grounding_paper_v2.models import TargetReview
from notepad_grounding_paper_v2.prompts import build_cell_choice_prompt
from notepad_grounding_paper_v2.prompts import build_desktop_review_prompt
from notepad_grounding_paper_v2.prompts import build_target_review_prompt

DEFAULT_MODEL = "gpt-5.4"
CHOOSE_CELL_RETRIES = 2

logger = logging.getLogger(__name__)


def capture_desktop() -> Image.Image:
    with mss.MSS() as screen_capture:
        screenshot = screen_capture.grab(screen_capture.monitors[1])
    return Image.frombytes("RGB", screenshot.size, screenshot.rgb)


def _image_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _ask(prompt: str, image: Image.Image, response_model):
    response = OpenAI().responses.parse(
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": _image_url(image), "detail": "high"},
                ],
            }
        ],
        text_format=response_model,
    )
    if response.output_parsed is None:
        raise ValueError(f"LLM returned no parsed {response_model.__name__}")
    return response.output_parsed


def choose_cell(
    *, query: str, image: Image.Image, cell_ids: list[str], rejected=(), reviewer_rationale: str = "", style: str = "zoom"
) -> CellChoice:
    """Pick one grid cell. Retries in-prompt when the model answers with an invalid cell id."""
    valid = [cell_id for cell_id in cell_ids if cell_id not in set(rejected)]
    prompt = build_cell_choice_prompt(
        query=query, cell_ids=valid, style=style, rejected=rejected, reviewer_rationale=reviewer_rationale
    )
    for attempt in range(CHOOSE_CELL_RETRIES + 1):
        choice = _ask(prompt, image, CellChoice)
        if choice.cell_id in valid:
            return choice
        logger.warning("attempt %d returned invalid cell_id %r; valid: %s", attempt + 1, choice.cell_id, valid)
        prompt += f"\nYour previous answer {choice.cell_id!r} was not a valid cell_id. Choose one of the valid values."
    raise ValueError(f"LLM failed to return a valid cell_id after {CHOOSE_CELL_RETRIES + 1} attempts")


def review_target(*, query: str, image: Image.Image) -> TargetReview:
    return _ask(build_target_review_prompt(query=query), image, TargetReview)


def review_desktop_state(*, action: str, expected: str, image: Image.Image) -> DesktopReview:
    return _ask(build_desktop_review_prompt(action=action, expected=expected), image, DesktopReview)
