from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict
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
OUT_DIR = Path("output/direct_llm_grounding")


@dataclass(frozen=True)
class CoordinateGuess:
    x: int
    y: int
    bbox: tuple[int, int, int, int]
    confidence: float
    rationale: str


def run() -> CoordinateGuess:
    load_env_file()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Add it to .env or set it in the shell.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    screenshot = capture_desktop()
    screenshot_path = OUT_DIR / "screenshot.png"
    screenshot.save(screenshot_path)

    raw_response = ask_llm_for_coordinates(screenshot)
    (OUT_DIR / "response.txt").write_text(raw_response, encoding="utf-8")

    guess = parse_coordinate_guess(raw_response, image_size=screenshot.size)
    (OUT_DIR / "result.json").write_text(json.dumps(asdict(guess), indent=2) + "\n", encoding="utf-8")

    annotated = draw_debug_box(screenshot, guess)
    annotated.save(OUT_DIR / "annotated.png")
    print(f"screenshot: {screenshot_path}")
    print(f"response:   {OUT_DIR / 'response.txt'}")
    print(f"annotated:  {OUT_DIR / 'annotated.png'}")
    print(json.dumps(asdict(guess), indent=2))
    return guess


def capture_desktop() -> Image.Image:
    with mss.MSS() as screen_capture:
        screenshot = screen_capture.grab(screen_capture.monitors[1])
    return Image.frombytes("RGB", screenshot.size, screenshot.rgb)


def ask_llm_for_coordinates(image: Image.Image) -> str:
    width, height = image.size
    prompt = f"""
You are looking at a Windows desktop screenshot.

The screenshot dimensions are exactly width={width} pixels and height={height} pixels.
The coordinate system starts at (0, 0) in the top-left corner.
x increases to the right. y increases downward.

Find the desktop shortcut icon for: {QUERY}

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


def parse_bbox(
    value: object,
    *,
    center: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    if isinstance(value, dict):
        if {"left", "top", "right", "bottom"} <= set(value):
            left = round(float(value["left"]))
            top = round(float(value["top"]))
            right = round(float(value["right"]))
            bottom = round(float(value["bottom"]))
        elif {"x", "y", "width", "height"} <= set(value):
            left = round(float(value["x"]))
            top = round(float(value["y"]))
            right = left + round(float(value["width"]))
            bottom = top + round(float(value["height"]))
        else:
            left, top, right, bottom = default_bbox(center)
    else:
        left, top, right, bottom = default_bbox(center)

    left = clamp(left, 0, width - 1)
    top = clamp(top, 0, height - 1)
    right = clamp(right, 0, width - 1)
    bottom = clamp(bottom, 0, height - 1)
    if right <= left:
        right = min(width - 1, left + 1)
    if bottom <= top:
        bottom = min(height - 1, top + 1)
    return left, top, right, bottom


def draw_debug_box(image: Image.Image, guess: CoordinateGuess) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    red = (255, 0, 0)
    left, top, right, bottom = guess.bbox

    draw.rectangle((left, top, right, bottom), outline=red, width=4)
    draw.line((guess.x - 16, guess.y, guess.x + 16, guess.y), fill=red, width=3)
    draw.line((guess.x, guess.y - 16, guess.x, guess.y + 16), fill=red, width=3)

    label = f"{QUERY} ({guess.x}, {guess.y}) conf={guess.confidence:.2f}"
    label_left = left
    label_top = max(0, top - 18)
    text_box = draw.textbbox((label_left, label_top), label, font=font)
    draw.rectangle(
        (text_box[0] - 3, text_box[1] - 2, text_box[2] + 3, text_box[3] + 2),
        fill=(255, 255, 255),
    )
    draw.text((label_left, label_top), label, fill=red, font=font)
    return annotated


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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


def default_bbox(center: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y = center
    half_size = 40
    return x - half_size, y - half_size, x + half_size, y + half_size


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def load_env_file(path: Path | None = None) -> None:
    env_path = path or Path.cwd() / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = strip_quotes(value.strip())
        if key and key not in os.environ:
            os.environ[key] = value


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


if __name__ == "__main__":
    run()
