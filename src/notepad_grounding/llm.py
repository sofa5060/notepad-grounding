from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import mss
from openai import OpenAI
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

QUERY = "Notepad"
DEFAULT_MODEL = "gpt-5.4"


@dataclass(frozen=True)
class CoordinateGuess:
    x: int
    y: int
    bbox: tuple[int, int, int, int]
    confidence: float
    rationale: str


def locate_icon(*, query: str = QUERY, output_dir: Path | None = None) -> CoordinateGuess:
    screenshot = capture_desktop()
    raw_response = ask_llm_for_coordinates(screenshot, query=query)
    guess = parse_coordinate_guess(raw_response, image_size=screenshot.size)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        screenshot.save(output_dir / "screenshot.png")
        (output_dir / "response.txt").write_text(raw_response, encoding="utf-8")
        draw_debug_box(screenshot, guess, query=query).save(output_dir / "annotated.png")

    return guess


def capture_desktop() -> Image.Image:
    with mss.MSS() as screen_capture:
        screenshot = screen_capture.grab(screen_capture.monitors[1])
    return Image.frombytes("RGB", screenshot.size, screenshot.rgb)


def ask_llm_for_coordinates(image: Image.Image, *, query: str = QUERY) -> str:
    width, height = image.size
    prompt = f"""
You are looking at a Windows desktop screenshot.

The screenshot dimensions are exactly width={width} pixels and height={height} pixels.
The coordinate system starts at (0, 0) in the top-left corner.
x increases to the right. y increases downward.

Find the desktop shortcut icon for: {query}

Return your best estimate as JSON only:
{{
  "x": 0,
  "y": 0,
  "bbox": {{"left": 0, "top": 0, "right": 0, "bottom": 0}},
  "confidence": 0.0,
  "rationale": "short explanation"
}}

Use x and y for the center of the icon graphic. Use bbox for the visible icon graphic area.
Do not return Markdown. Do not explain outside the JSON.
""".strip()
    client = OpenAI()
    response = client.responses.create(
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
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
    return response.output_text


def parse_coordinate_guess(text: str, *, image_size: tuple[int, int]) -> CoordinateGuess:
    payload = json.loads(strip_json_fence(text))
    width, height = image_size
    x = clamp(round(float(payload["x"])), 0, width - 1)
    y = clamp(round(float(payload["y"])), 0, height - 1)
    bbox = parse_bbox(payload.get("bbox"), center=(x, y), image_size=image_size)
    confidence = clamp_float(float(payload.get("confidence", 0.0)), 0.0, 1.0)
    rationale = str(payload.get("rationale", "")).strip()
    return CoordinateGuess(x=x, y=y, bbox=bbox, confidence=confidence, rationale=rationale)


def parse_bbox(value: object, *, center: tuple[int, int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    if isinstance(value, dict) and {"left", "top", "right", "bottom"} <= set(value):
        left = round(float(value["left"]))
        top = round(float(value["top"]))
        right = round(float(value["right"]))
        bottom = round(float(value["bottom"]))
    else:
        x, y = center
        left, top, right, bottom = x - 40, y - 40, x + 40, y + 40

    left = clamp(left, 0, width - 1)
    top = clamp(top, 0, height - 1)
    right = clamp(right, 0, width - 1)
    bottom = clamp(bottom, 0, height - 1)
    return left, top, max(left + 1, right), max(top + 1, bottom)


def draw_debug_box(image: Image.Image, guess: CoordinateGuess, *, query: str = QUERY) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    red = (255, 0, 0)
    left, top, right, bottom = guess.bbox

    draw.rectangle((left, top, right, bottom), outline=red, width=4)
    draw.line((guess.x - 16, guess.y, guess.x + 16, guess.y), fill=red, width=3)
    draw.line((guess.x, guess.y - 16, guess.x, guess.y + 16), fill=red, width=3)

    label = f"{query} ({guess.x}, {guess.y}) conf={guess.confidence:.2f}"
    text_box = draw.textbbox((left, max(0, top - 18)), label, font=font)
    draw.rectangle((text_box[0] - 3, text_box[1] - 2, text_box[2] + 3, text_box[3] + 2), fill=(255, 255, 255))
    draw.text((left, max(0, top - 18)), label, fill=red, font=font)
    return annotated


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))
