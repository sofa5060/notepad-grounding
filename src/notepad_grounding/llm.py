from __future__ import annotations
import base64
import logging
import os
from io import BytesIO
from pathlib import Path
import mss
from openai import OpenAI
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from notepad_grounding.models import AppOpenReview
from notepad_grounding.models import IconBBox
from notepad_grounding.models import IconLocation
from notepad_grounding.prompts import build_app_open_review_prompt
from notepad_grounding.prompts import build_locate_prompt

DEFAULT_MODEL = "gpt-5.4"
MIN_CONFIDENCE = 0.85

logger = logging.getLogger(__name__)


def capture_desktop() -> Image.Image:
    with mss.MSS() as screen_capture:
        screenshot = screen_capture.grab(screen_capture.monitors[1])
    return Image.frombytes("RGB", screenshot.size, screenshot.rgb)


def _image_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def locate_icon(*, query: str, output_dir: Path) -> IconLocation | None:
    image = capture_desktop()
    (width, height) = image.size

    prompt = build_locate_prompt(query=query, width=width, height=height)

    try:
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
            text_format=IconLocation,
        )
        raw = response.output_parsed
    except Exception as exc:
        logger.warning("LLM locate attempt failed: %s", exc)
        return None
    if raw is None:
        logger.warning("LLM locate attempt returned no parsed result")
        return None

    if raw.confidence < MIN_CONFIDENCE:
        logger.warning("confidence below threshold (%.2f < %.2f)", raw.confidence, MIN_CONFIDENCE)
        return None

    x = max(0, min(raw.x, width - 1))
    y = max(0, min(raw.y, height - 1))
    left = max(0, min(raw.bbox.left, width - 1))
    top = max(0, min(raw.bbox.top, height - 1))
    right = max(left, min(raw.bbox.right, width - 1))
    bottom = max(top, min(raw.bbox.bottom, height - 1))

    result = IconLocation(
        x=x,
        y=y,
        bbox=IconBBox(left=left, top=top, right=right, bottom=bottom),
        confidence=raw.confidence,
        rationale=raw.rationale.strip(),
    )

    save_artifacts(image=image, result=result, query=query, output_dir=output_dir)

    return result


def verify_app_opened(
    *, expected_app: str, before_image: Image.Image, after_image: Image.Image
) -> AppOpenReview | None:
    prompt = build_app_open_review_prompt(expected_app=expected_app)
    try:
        response = OpenAI().responses.parse(
            model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_text", "text": "Before screenshot:"},
                        {"type": "input_image", "image_url": _image_url(before_image), "detail": "high"},
                        {"type": "input_text", "text": "After screenshot:"},
                        {"type": "input_image", "image_url": _image_url(after_image), "detail": "high"},
                    ],
                }
            ],
            text_format=AppOpenReview,
        )
    except Exception as exc:
        logger.warning("LLM app-open verification failed: %s", exc)
        return None

    if response.output_parsed is None:
        logger.warning("LLM app-open verification returned no parsed result")
    return response.output_parsed


def save_artifacts(*, image: Image.Image, result: IconLocation, query: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / "screenshot.png")
    (output_dir / "response.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")

    b = result.bbox
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    red = (255, 0, 0)
    draw.rectangle((b.left, b.top, b.right, b.bottom), outline=red, width=4)
    draw.line((result.x - 16, result.y, result.x + 16, result.y), fill=red, width=3)
    draw.line((result.x, result.y - 16, result.x, result.y + 16), fill=red, width=3)
    label = f"{query} ({result.x}, {result.y}) conf={result.confidence:.2f}"
    text_box = draw.textbbox((b.left, max(0, b.top - 18)), label, font=font)
    draw.rectangle((text_box[0] - 3, text_box[1] - 2, text_box[2] + 3, text_box[3] + 2), fill=(255, 255, 255))
    draw.text((b.left, max(0, b.top - 18)), label, fill=red, font=font)
    annotated.save(output_dir / "annotated.png")
