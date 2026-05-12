from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from notepad_grounding.grounding import Candidate
from notepad_grounding.grounding import GridCell
from notepad_grounding.ocr import Box
from notepad_grounding.ocr import OcrLine


def draw_grid(image: Image.Image, grid: Iterable[GridCell], output_path: Path) -> Path:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for cell in grid:
        draw.rectangle(cell.box, outline=(190, 190, 190), width=1)
        if cell.row % 2 == 0 and cell.col % 4 == 0:
            draw.text((cell.box[0] + 3, cell.box[1] + 3), f"{cell.row},{cell.col}", fill=(90, 90, 90), font=font)
    return _save(annotated, output_path)


def draw_ocr(image: Image.Image, lines: Iterable[OcrLine], output_path: Path) -> Path:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for line in lines:
        _draw_box(draw, line.box, (0, 180, 80), line.text, font)
    return _save(annotated, output_path)


def draw_candidates(
    image: Image.Image,
    *,
    candidates: Iterable[Candidate],
    selected: Candidate | None,
    output_path: Path,
) -> Path:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for candidate in candidates:
        label = f"#{candidate.id} {candidate.label_text} {candidate.score:.2f}"
        _draw_box(draw, candidate.combined_box, (255, 210, 0), label, font)
        _draw_box(draw, candidate.icon_box, (0, 120, 255), "icon", font)

    if selected is not None:
        _draw_box(draw, selected.icon_box, (255, 0, 0), "selected", font, width=4)
        x, y = selected.icon_center
        draw.line((x - 10, y, x + 10, y), fill=(255, 0, 0), width=3)
        draw.line((x, y - 10, x, y + 10), fill=(255, 0, 0), width=3)
    return _save(annotated, output_path)


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: Box,
    color: tuple[int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    width: int = 2,
) -> None:
    draw.rectangle(box, outline=color, width=width)
    x = box[0] + 3
    y = max(0, box[1] - 14)
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(255, 255, 255))
    draw.text((x, y), text, fill=color, font=font)


def _save(image: Image.Image, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path
