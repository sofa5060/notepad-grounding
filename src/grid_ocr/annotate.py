from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from grid_ocr.grounding import Candidate
from grid_ocr.ocr import Box
from grid_ocr.ocr import OcrLine

_FONT = ImageFont.load_default()


def draw_ocr(image: Image.Image, lines: Iterable[OcrLine], output_path: Path) -> Path:
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for line in lines:
        _draw_box(draw, line.box, (0, 180, 80), line.text)
    return _save(annotated, output_path)


def draw_candidates(
    image: Image.Image, *, candidates: Iterable[Candidate], selected: Candidate | None, output_path: Path
) -> Path:
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for number, candidate in enumerate(candidates, start=1):
        label = f"#{number} {candidate.label_text} {candidate.score:.2f}"
        _draw_box(draw, candidate.combined_box, (255, 210, 0), label)
        _draw_box(draw, candidate.icon_box, (0, 120, 255), "icon")

    if selected is not None:
        _draw_box(draw, selected.icon_box, (255, 0, 0), "selected", width=4)
        (x, y) = selected.icon_center
        draw.line((x - 10, y, x + 10, y), fill=(255, 0, 0), width=3)
        draw.line((x, y - 10, x, y + 10), fill=(255, 0, 0), width=3)
    return _save(annotated, output_path)


def _draw_box(draw: ImageDraw.ImageDraw, box: Box, color: tuple[int, int, int], text: str, *, width: int = 2) -> None:
    draw.rectangle(box, outline=color, width=width)
    x = box[0] + 3
    y = max(0, box[1] - 14)
    (left, top, right, bottom) = draw.textbbox((x, y), text, font=_FONT)
    draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(255, 255, 255))
    draw.text((x, y), text, fill=color, font=_FONT)


def _save(image: Image.Image, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path
