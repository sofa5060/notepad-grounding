from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from notepad_grounding_paper_v2 import llm

FIRST_GRID = (3, 4)
LATER_GRID = (3, 3)
ZOOM_ROUNDS = 3
CROP_PADDING = 40
FINAL_CROP_MAX_SIZE = (450, 350)
COARSE_CLICK_GRID = (7, 7)
FINE_CLICK_GRID = (5, 5)
FINE_CROP_SIZE = (96, 96)
MAX_REVIEW_RETRIES = 2

Box = tuple[int, int, int, int]

_FONT = ImageFont.load_default()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridCell:
    id: str
    row: int
    col: int
    box: Box

    @property
    def center(self) -> tuple[int, int]:
        return ((self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2)


def build_grid_cells(box: Box, *, rows: int, cols: int) -> list[GridCell]:
    (x1, y1, x2, y2) = box
    (width, height) = (x2 - x1, y2 - y1)
    cells: list[GridCell] = []
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            cell_box = (
                x1 + round(width * (col - 1) / cols),
                y1 + round(height * (row - 1) / rows),
                x1 + round(width * col / cols),
                y1 + round(height * row / rows),
            )
            cells.append(GridCell(id=f"R{row}C{col}", row=row, col=col, box=cell_box))
    return cells


def cell_by_id(cells, cell_id: str) -> GridCell:
    for cell in cells:
        if cell.id == cell_id:
            return cell
    raise ValueError(f"Unknown cell: {cell_id}")


def offset_box(box: Box, *, offset: tuple[int, int]) -> Box:
    return (box[0] + offset[0], box[1] + offset[1], box[2] + offset[0], box[3] + offset[1])


def expand_box(box: Box, *, padding: int, bounds: Box) -> Box:
    return (
        max(bounds[0], box[0] - padding),
        max(bounds[1], box[1] - padding),
        min(bounds[2], box[2] + padding),
        min(bounds[3], box[3] + padding),
    )


def crop_around_point(image: Image.Image, *, center: tuple[int, int], size: tuple[int, int]) -> tuple[Image.Image, Box]:
    (width, height) = image.size
    (crop_width, crop_height) = (min(size[0], width), min(size[1], height))
    x1 = max(0, min(center[0] - crop_width // 2, width - crop_width))
    y1 = max(0, min(center[1] - crop_height // 2, height - crop_height))
    box: Box = (x1, y1, x1 + crop_width, y1 + crop_height)
    return (image.crop(box), box)


def _save(image: Image.Image, output_path: Path) -> Image.Image:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return image


def draw_grid(image: Image.Image, cells, *, output_path: Path, selected_cell_id: str | None = None) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for cell in cells:
        selected = cell.id == selected_cell_id
        color = (255, 0, 0) if selected else (255, 210, 0)
        draw.rectangle(cell.box, outline=color, width=4 if selected else 2)
        label_anchor = (cell.box[0] + 4, cell.box[1] + 4)
        (left, top, right, bottom) = draw.textbbox(label_anchor, cell.id, font=_FONT)
        draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(255, 255, 255))
        draw.text(label_anchor, cell.id, fill=color, font=_FONT)
    return _save(annotated, output_path)


def draw_click_grid(
    image: Image.Image, cells, *, output_path: Path, selected_cell_id: str | None = None, gutter: int = 28
) -> Image.Image:
    source = image.convert("RGB")
    annotated = Image.new("RGB", (source.width + gutter, source.height + gutter), (255, 255, 255))
    annotated.paste(source, (gutter, gutter))
    draw = ImageDraw.Draw(annotated)
    cells = list(cells)
    for col in range(1, max(cell.col for cell in cells) + 1):
        matching = [cell for cell in cells if cell.col == col]
        x1 = min(cell.box[0] for cell in matching) + gutter
        x2 = max(cell.box[2] for cell in matching) + gutter
        _draw_centered_label(draw, str(col), x_range=(x1, x2), y_range=(0, gutter))
    for row in range(1, max(cell.row for cell in cells) + 1):
        matching = [cell for cell in cells if cell.row == row]
        y1 = min(cell.box[1] for cell in matching) + gutter
        y2 = max(cell.box[3] for cell in matching) + gutter
        _draw_centered_label(draw, str(row), x_range=(0, gutter), y_range=(y1, y2))
    for cell in cells:
        box = (cell.box[0] + gutter, cell.box[1] + gutter, cell.box[2] + gutter, cell.box[3] + gutter)
        if cell.id == selected_cell_id:
            draw.rectangle(box, outline=(255, 210, 0), width=3)
        else:
            draw.rectangle(box, outline=(255, 0, 0), width=1)
    return _save(annotated, output_path)


def _draw_centered_label(draw: ImageDraw.ImageDraw, label: str, *, x_range, y_range) -> None:
    (left, top, right, bottom) = draw.textbbox((0, 0), label, font=_FONT)
    x = (x_range[0] + x_range[1] - (right - left)) // 2
    y = (y_range[0] + y_range[1] - (bottom - top)) // 2
    draw.text((x, y), label, fill=(0, 0, 0), font=_FONT)


def draw_click_marker(image: Image.Image, *, point: tuple[int, int], output_path: Path) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    (x, y) = point
    red = (255, 0, 0)
    draw.line((x - 14, y, x + 14, y), fill=red, width=5)
    draw.line((x, y - 14, x, y + 14), fill=red, width=5)
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=red, width=4)
    return _save(annotated, output_path)
