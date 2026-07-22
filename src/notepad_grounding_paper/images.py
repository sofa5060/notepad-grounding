"""Grid geometry, cropping, and drawing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

Box = tuple[int, int, int, int]

_FONT = ImageFont.load_default()


@dataclass(frozen=True)
class GridCell:
    id: str
    row: int
    col: int
    box: Box

    @property
    def center(self) -> tuple[int, int]:
        return ((self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2)


def build_grid_cells(box: Box, *, rows: int, cols: int, prefix: str = "A", id_fmt: str = "dash") -> list[GridCell]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    (x1, y1, x2, y2) = box
    width = x2 - x1
    height = y2 - y1
    cells: list[GridCell] = []
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            cell_x1 = x1 + round(width * (col - 1) / cols)
            cell_y1 = y1 + round(height * (row - 1) / rows)
            cell_x2 = x1 + round(width * col / cols)
            cell_y2 = y1 + round(height * row / rows)
            if id_fmt == "rc":
                cell_id = f"R{row}C{col}"
            else:
                cell_id = f"{prefix}{row}-{col}"
            cells.append(GridCell(id=cell_id, row=row, col=col, box=(cell_x1, cell_y1, cell_x2, cell_y2)))
    return cells


def cell_by_id(cells: Iterable[GridCell], cell_id: str) -> GridCell:
    for cell in cells:
        if cell.id == cell_id:
            return cell
    raise ValueError(f"Unknown cell: {cell_id}")


def offset_point(point: tuple[int, int], *, offset: tuple[int, int]) -> tuple[int, int]:
    return (point[0] + offset[0], point[1] + offset[1])


def offset_box(box: Box, *, offset: tuple[int, int]) -> Box:
    return (box[0] + offset[0], box[1] + offset[1], box[2] + offset[0], box[3] + offset[1])


def clamp_box(box: Box, bounds: Box) -> Box:
    return (
        max(bounds[0], min(box[0], bounds[2])),
        max(bounds[1], min(box[1], bounds[3])),
        max(bounds[0], min(box[2], bounds[2])),
        max(bounds[1], min(box[3], bounds[3])),
    )


def expand_box(box: Box, *, padding: int, bounds: Box) -> Box:
    return clamp_box((box[0] - padding, box[1] - padding, box[2] + padding, box[3] + padding), bounds)


def crop_around_point(image: Image.Image, *, center: tuple[int, int], size: tuple[int, int]) -> tuple[Image.Image, Box]:
    (width, height) = image.size
    crop_width = min(size[0], width)
    crop_height = min(size[1], height)
    x1 = max(0, min(center[0] - crop_width // 2, width - crop_width))
    y1 = max(0, min(center[1] - crop_height // 2, height - crop_height))
    box: Box = (x1, y1, x1 + crop_width, y1 + crop_height)
    return (image.crop(box), box)


def _save(image: Image.Image, output_path: Path) -> Image.Image:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return image


def draw_grid_cells(
    image: Image.Image, cells: Iterable[GridCell], *, output_path: Path, selected_cell_id: str | None = None
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for cell in cells:
        is_selected = cell.id == selected_cell_id
        color = (255, 0, 0) if is_selected else (255, 210, 0)
        width = 4 if is_selected else 2
        draw.rectangle(cell.box, outline=color, width=width)
        label_x = cell.box[0] + 4
        label_y = cell.box[1] + 4
        (left, top, right, bottom) = draw.textbbox((label_x, label_y), cell.id, font=_FONT)
        draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(255, 255, 255))
        draw.text((label_x, label_y), cell.id, fill=color, font=_FONT)
    return _save(annotated, output_path)


def draw_click_grid(
    image: Image.Image,
    cells: Iterable[GridCell],
    *,
    output_path: Path,
    selected_cell_id: str | None = None,
    gutter: int = 28,
) -> Image.Image:
    source = image.convert("RGB")
    annotated = Image.new("RGB", (source.width + gutter, source.height + gutter), (255, 255, 255))
    annotated.paste(source, (gutter, gutter))
    draw = ImageDraw.Draw(annotated)
    cells_list = list(cells)
    rows = max(cell.row for cell in cells_list)
    cols = max(cell.col for cell in cells_list)

    for col in range(1, cols + 1):
        matching = [cell for cell in cells_list if cell.col == col]
        x1 = min(cell.box[0] for cell in matching) + gutter
        x2 = max(cell.box[2] for cell in matching) + gutter
        label = str(col)
        (left, top, right, bottom) = draw.textbbox((0, 0), label, font=_FONT)
        draw.text(((x1 + x2 - (right - left)) // 2, (gutter - (bottom - top)) // 2), label, fill=(0, 0, 0), font=_FONT)

    for row in range(1, rows + 1):
        matching = [cell for cell in cells_list if cell.row == row]
        y1 = min(cell.box[1] for cell in matching) + gutter
        y2 = max(cell.box[3] for cell in matching) + gutter
        label = str(row)
        (left, top, right, bottom) = draw.textbbox((0, 0), label, font=_FONT)
        draw.text(((gutter - (right - left)) // 2, (y1 + y2 - (bottom - top)) // 2), label, fill=(0, 0, 0), font=_FONT)

    for cell in cells_list:
        box = (cell.box[0] + gutter, cell.box[1] + gutter, cell.box[2] + gutter, cell.box[3] + gutter)
        if cell.id == selected_cell_id:
            draw.rectangle(box, outline=(255, 210, 0), width=3)
        else:
            draw.rectangle(box, outline=(255, 0, 0), width=1)

    return _save(annotated, output_path)


def draw_full_click_marker(
    image: Image.Image, *, point: tuple[int, int], output_path: Path, label: str = "click_point"
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    (x, y) = point
    color = (255, 0, 0)

    draw.line((x - 14, y, x + 14, y), fill=color, width=5)
    draw.line((x, y - 14, x, y + 14), fill=color, width=5)
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=4)

    (left, top, right, bottom) = draw.textbbox((0, 0), label, font=_FONT)
    label_image = Image.new("RGBA", (right - left + 6, bottom - top + 4), (255, 255, 255, 230))
    ImageDraw.Draw(label_image).text((3, 2), label, fill=color, font=_FONT)
    label_image = label_image.resize((label_image.width * 2, label_image.height * 2))
    label_x = min(max(0, x + 18), max(0, annotated.width - label_image.width))
    if y - label_image.height - 18 >= 0:
        label_y = y - label_image.height - 18
    else:
        label_y = min(y + 18, max(0, annotated.height - label_image.height))
    annotated.paste(label_image.convert("RGB"), (label_x, label_y))

    return _save(annotated, output_path)
