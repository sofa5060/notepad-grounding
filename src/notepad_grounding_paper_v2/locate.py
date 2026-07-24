from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from notepad_grounding_paper_v2 import llm

ZOOM_GRID = (3, 4)
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


def _choose_reviewed_cell(*, crop, cells, query, style, output_dir, tag, max_retries=MAX_REVIEW_RETRIES) -> GridCell:
    """One grid round: the LLM picks a cell, the reviewer must approve the highlighted pick.

    On rejection, re-asks with the rejected cells excluded and the reviewer's rationale in the prompt.
    Raises ValueError when the reviewer rejects every attempt.
    """
    draw = draw_grid if style == "zoom" else draw_click_grid
    overlay = draw(crop, cells, output_path=output_dir / f"{tag}-grid.png")
    cell_ids = [cell.id for cell in cells]
    (rejected, rationale) = ([], "")
    for _ in range(max_retries + 1):
        choice = llm.choose_cell(
            query=query, image=overlay, cell_ids=cell_ids, rejected=rejected, reviewer_rationale=rationale, style=style
        )
        highlighted = draw(crop, cells, output_path=output_dir / f"{tag}-selected.png", selected_cell_id=choice.cell_id)
        review = llm.review_target(query=query, image=highlighted)
        if review.contains_target:
            logger.info("%s: chose cell %s (confidence %.2f)", tag, choice.cell_id, choice.confidence)
            return cell_by_id(cells, choice.cell_id)
        rejected.append(choice.cell_id)
        rationale = review.rationale
        logger.info("%s: reviewer rejected %s: %s", tag, choice.cell_id, review.rationale)
    raise ValueError(f"Target reviewer rejected all attempts for {tag}; rejected={rejected}; last_rationale={rationale}")


def run_locate(image: Image.Image, *, query: str, output_root: Path, timestamp: str | None = None, rounds: int = ZOOM_ROUNDS) -> tuple[int, int]:
    """Find the screen coordinates of `query` (e.g. the Notepad icon) in `image`.

    Zooms in with LLM-guided, reviewer-approved grid choices, then refines to an
    exact pixel with a coarse and a fine click grid. Saves artifacts per stage.
    """
    start = time.perf_counter()
    run_id = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_dir / "00-source.png")

    bounds: Box = (0, 0, image.width, image.height)
    current_box = bounds
    for round_index in range(1, rounds + 1):
        (width, height) = (current_box[2] - current_box[0], current_box[3] - current_box[1])
        if width <= FINAL_CROP_MAX_SIZE[0] and height <= FINAL_CROP_MAX_SIZE[1]:
            break
        crop = image.crop(current_box)
        cells = build_grid_cells((0, 0, crop.width, crop.height), rows=ZOOM_GRID[0], cols=ZOOM_GRID[1])
        cell = _choose_reviewed_cell(
            crop=crop, cells=cells, query=query, style="zoom", output_dir=output_dir, tag=f"{round_index:02d}"
        )
        selected = offset_box(cell.box, offset=current_box[:2])
        current_box = expand_box(selected, padding=CROP_PADDING, bounds=bounds)

    final_crop = image.crop(current_box)
    final_crop.save(output_dir / "final-crop.png")

    coarse = _choose_reviewed_cell(
        crop=final_crop,
        cells=build_grid_cells((0, 0, final_crop.width, final_crop.height), rows=COARSE_CLICK_GRID[0], cols=COARSE_CLICK_GRID[1]),
        query=query,
        style="click",
        output_dir=output_dir,
        tag="click-01",
    )
    (fine_crop, fine_box) = crop_around_point(final_crop, center=coarse.center, size=FINE_CROP_SIZE)
    fine = _choose_reviewed_cell(
        crop=fine_crop,
        cells=build_grid_cells((0, 0, fine_crop.width, fine_crop.height), rows=FINE_CLICK_GRID[0], cols=FINE_CLICK_GRID[1]),
        query=query,
        style="click",
        output_dir=output_dir,
        tag="click-02",
    )

    point_in_final = (fine.center[0] + fine_box[0], fine.center[1] + fine_box[1])
    screen_point = (point_in_final[0] + current_box[0], point_in_final[1] + current_box[1])
    draw_click_marker(image, point=screen_point, output_path=output_dir / "click-point-full.png")
    logger.info("located %r at %s in %.2fs; artifacts in %s", query, screen_point, time.perf_counter() - start, output_dir)
    return screen_point
