from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str


def default_reference_boxes(width: int, height: int) -> list[Box]:
    box_size = 96
    half = box_size // 2
    right = max(width - 1, 0)
    bottom = max(height - 1, 0)
    center_x = width // 2
    center_y = height // 2

    return [
        Box(0, 0, min(box_size, right), min(box_size, bottom), "top-left origin"),
        Box(max(right - box_size, 0), 0, right, min(box_size, bottom), "top-right"),
        Box(0, max(bottom - box_size, 0), min(box_size, right), bottom, "bottom-left"),
        Box(
            max(center_x - half, 0),
            max(center_y - half, 0),
            min(center_x + half, right),
            min(center_y + half, bottom),
            "screen center",
        ),
    ]


def annotate_screenshot(
    image: Image.Image,
    *,
    boxes: Iterable[Box],
    output_path: Path,
    title: str | None = None,
    notes: Iterable[str] = (),
) -> Path:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for box in boxes:
        coords = _clamp_box(box, annotated.width, annotated.height)
        draw.rectangle(coords, outline=(255, 0, 0), width=3)
        _draw_label(draw, coords[0], coords[1], box.label, font)

    if title or notes:
        _draw_metadata(draw, title=title, notes=list(notes), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    return output_path


def annotate_candidate_proof(
    image: Image.Image,
    *,
    labels: Iterable[object],
    candidates: Iterable[object],
    output_path: Path,
    title: str | None = "Milestone 3 OCR candidate proof",
    notes: Iterable[str] = (),
) -> Path:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    yellow = (255, 210, 0)
    blue = (0, 120, 255)
    green = (0, 180, 80)

    for candidate in candidates:
        candidate_id = getattr(candidate, "candidate_id")
        label_text = getattr(candidate, "label_text")
        combined_box = getattr(candidate, "combined_box")
        icon_box = getattr(candidate, "icon_box")
        _draw_box(draw, combined_box, yellow, f"#{candidate_id} {label_text}", font)
        _draw_box(draw, icon_box, blue, "icon", font)

    for label in labels:
        label_box = getattr(label, "box")
        label_text = getattr(label, "text")
        _draw_box(draw, label_box, green, label_text, font)

    if title or notes:
        _draw_metadata(draw, title=title, notes=list(notes), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    return output_path


def annotate_ocr_proof(
    image: Image.Image,
    *,
    words: Iterable[object],
    lines: Iterable[object],
    output_path: Path,
    draw_words: bool = False,
    title: str | None = "OCR text proof",
    notes: Iterable[str] = (),
) -> Path:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    orange = (255, 150, 0)
    green = (0, 180, 80)

    if draw_words:
        for word in words:
            _draw_box(draw, getattr(word, "box"), orange, getattr(word, "text"), font)

    for index, line in enumerate(lines, start=1):
        text = getattr(line, "text")
        _draw_box(draw, getattr(line, "box"), green, f"#{index} {text}", font)

    if title or notes:
        _draw_metadata(draw, title=title, notes=list(notes), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    return output_path


def _clamp_box(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    x1 = min(max(box.x1, 0), max_x)
    y1 = min(max(box.y1, 0), max_y)
    x2 = min(max(box.x2, x1), max_x)
    y2 = min(max(box.y2, y1), max_y)
    return (x1, y1, x2, y2)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    label_x = x + 4
    label_y = max(y - 14, 0)
    left, top, right, bottom = draw.textbbox((label_x, label_y), text, font=font)
    draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(255, 255, 255))
    draw.text((label_x, label_y), text, fill=(255, 0, 0), font=font)


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: Box,
    color: tuple[int, int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    coords = _clamp_box(box, draw.im.size[0], draw.im.size[1])
    draw.rectangle(coords, outline=color, width=3)
    label_x = coords[0] + 4
    label_y = max(coords[1] - 14, 0)
    left, top, right, bottom = draw.textbbox((label_x, label_y), text, font=font)
    draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(255, 255, 255))
    draw.text((label_x, label_y), text, fill=color, font=font)


def _draw_metadata(
    draw: ImageDraw.ImageDraw,
    *,
    title: str | None,
    notes: list[str],
    font: ImageFont.ImageFont,
) -> None:
    lines = [line for line in [title, *notes] if line]
    if not lines:
        return

    line_height = 14
    width = max(draw.textlength(line, font=font) for line in lines)
    panel = (8, 8, int(width) + 20, 12 + line_height * len(lines))
    draw.rectangle(panel, fill=(255, 255, 255), outline=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((14, 12 + index * line_height), line, fill=(0, 0, 0), font=font)
