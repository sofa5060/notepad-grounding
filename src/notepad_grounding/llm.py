from __future__ import annotations
import base64
import logging
import os
import time
from io import BytesIO
from pathlib import Path
import mss
from openai import OpenAI
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from pydantic import BaseModel, Field

DEFAULT_MODEL = "gpt-5.4"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0
MIN_CONFIDENCE = 0.85

logger = logging.getLogger(__name__)


class IconBBox(BaseModel):
    left: int = Field(description="Left edge of the icon bounding box in pixels")
    top: int = Field(description="Top edge of the icon bounding box in pixels")
    right: int = Field(description="Right edge of the icon bounding box in pixels")
    bottom: int = Field(description="Bottom edge of the icon bounding box in pixels")


class IconLocation(BaseModel):
    x: int = Field(description="X coordinate of the icon center in pixels")
    y: int = Field(description="Y coordinate of the icon center in pixels")
    bbox: IconBBox = Field(description="Bounding box of the visible icon graphic area")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    rationale: str = Field(description="Short explanation of the location choice")


def locate_icon(*, query: str, output_dir: Path) -> IconLocation | None:
    with mss.MSS() as screen_capture:
        screenshot = screen_capture.grab(screen_capture.monitors[1])
    image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
    width, height = image.size

    prompt = f"""
        You are looking at a Windows desktop screenshot.

        The screenshot dimensions are exactly width={width} pixels and height={height} pixels.
        The coordinate system starts at (0, 0) in the top-left corner.
        x increases to the right. y increases downward.

        Find the desktop shortcut icon for: {query}

        Use x and y for the center of the icon graphic. Use bbox for the visible icon graphic area.
    """.strip()

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    image_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"

    raw = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = OpenAI().responses.parse(
                model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_url, "detail": "high"},
                        ],
                    }
                ],
                text_format=IconLocation,
            )
            raw = response.output_parsed
        except Exception as exc:
            logger.warning("attempt %d/%d: API error — %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            continue

        logger.info("attempt %d/%d: confidence=%.2f", attempt, MAX_RETRIES, raw.confidence)
        if raw.confidence >= MIN_CONFIDENCE:
            break

        if attempt < MAX_RETRIES:
            logger.info("confidence below threshold (%.2f < %.2f), retrying in %ds",
                        raw.confidence, MIN_CONFIDENCE, RETRY_DELAY_SECONDS)
            time.sleep(RETRY_DELAY_SECONDS)

    if raw is None:
        raise RuntimeError(f"Failed to locate {query} after {MAX_RETRIES} attempts")

    if raw.confidence < MIN_CONFIDENCE:
        logger.warning("all %d attempts below confidence threshold (best=%.2f < %.2f)",
                       MAX_RETRIES, raw.confidence, MIN_CONFIDENCE)
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
