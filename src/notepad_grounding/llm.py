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

DEFAULT_MODEL = "gpt-5.4"


@dataclass(frozen=True)
class CoordinateGuess:
    x: int
    y: int
    bbox: tuple[int, int, int, int]
    confidence: float
    rationale: str


def locate_icon(*, query: str, output_dir: Path) -> CoordinateGuess:
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

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    image_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"
    response = OpenAI().responses.create(
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
    )

    raw_text = response.output_text
    json_text = raw_text.strip()
    if json_text.startswith("```"):
        lines = json_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        json_text = "\n".join(lines).strip()

    payload = json.loads(json_text)
    x = max(0, min(round(float(payload["x"])), width - 1))
    y = max(0, min(round(float(payload["y"])), height - 1))

    bbox_payload = payload.get("bbox")
    if isinstance(bbox_payload, dict) and {"left", "top", "right", "bottom"} <= set(bbox_payload):
        left = round(float(bbox_payload["left"]))
        top = round(float(bbox_payload["top"]))
        right = round(float(bbox_payload["right"]))
        bottom = round(float(bbox_payload["bottom"]))
    else:
        left, top, right, bottom = x - 40, y - 40, x + 40, y + 40

    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left, min(right, width - 1))
    bottom = max(top, min(bottom, height - 1))
    guess = CoordinateGuess(
        x=x,
        y=y,
        bbox=(left, top, right, bottom),
        confidence=max(0.0, min(float(payload.get("confidence", 0.0)), 1.0)),
        rationale=str(payload.get("rationale", "")).strip(),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / "screenshot.png")
    (output_dir / "response.txt").write_text(raw_text, encoding="utf-8")

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    red = (255, 0, 0)
    draw.rectangle(guess.bbox, outline=red, width=4)
    draw.line((guess.x - 16, guess.y, guess.x + 16, guess.y), fill=red, width=3)
    draw.line((guess.x, guess.y - 16, guess.x, guess.y + 16), fill=red, width=3)
    label = f"{query} ({guess.x}, {guess.y}) conf={guess.confidence:.2f}"
    text_box = draw.textbbox((left, max(0, top - 18)), label, font=font)
    draw.rectangle((text_box[0] - 3, text_box[1] - 2, text_box[2] + 3, text_box[3] + 2), fill=(255, 255, 255))
    draw.text((left, max(0, top - 18)), label, fill=red, font=font)
    annotated.save(output_dir / "annotated.png")

    return guess
