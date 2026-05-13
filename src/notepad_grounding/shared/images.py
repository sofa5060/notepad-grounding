from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from notepad_grounding.shared.geometry import Box
from notepad_grounding.shared.geometry import GridCell


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def crop_box(image: Image.Image, box: Box) -> Image.Image:
    return image.crop(box)


def draw_grid_cells(
    image: Image.Image,
    cells: Iterable[GridCell],
    *,
    output_path: Path,
    selected_cell_id: str | None = None,
    selected_cell_ids: list[str] | None = None,
) -> Path:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    selected_set = set(selected_cell_ids or [])
    if selected_cell_id:
        selected_set.add(selected_cell_id)
    for cell in cells:
        is_selected = cell.id in selected_set
        color = (255, 0, 0) if is_selected else (255, 210, 0)
        width = 4 if is_selected else 2
        draw.rectangle(cell.box, outline=color, width=width)
        label_x = cell.box[0] + 4
        label_y = cell.box[1] + 4
        left, top, right, bottom = draw.textbbox((label_x, label_y), cell.id, font=font)
        draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(255, 255, 255))
        draw.text((label_x, label_y), cell.id, fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    return output_path


def draw_box(
    image: Image.Image,
    box: Box,
    *,
    output_path: Path,
    label: str,
    color: tuple[int, int, int] = (255, 0, 0),
) -> Path:
    annotated = draw_box_on_image(image, box, label=label, color=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    return output_path


def draw_box_on_image(
    image: Image.Image,
    box: Box,
    *,
    label: str | None = None,
    color: tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Draw a bounding box on a copy of the image and return it (does not save)."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    draw.rectangle(box, outline=color, width=3)
    if label:
        left, top, right, bottom = draw.textbbox((box[0] + 4, max(0, box[1] - 14)), label, font=font)
        draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(255, 255, 255))
        draw.text((box[0] + 4, max(0, box[1] - 14)), label, fill=color, font=font)
    return annotated
