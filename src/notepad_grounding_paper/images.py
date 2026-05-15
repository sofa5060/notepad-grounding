from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from notepad_grounding_paper.geometry import Box
from notepad_grounding_paper.geometry import GridCell
from notepad_grounding_paper.geometry import cell_by_id


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def crop_box(image: Image.Image, box: Box) -> Image.Image:
    return image.crop(box)


def crop_around_point(
    image: Image.Image,
    *,
    center: tuple[int, int],
    size: tuple[int, int],
) -> tuple[Image.Image, Box]:
    width, height = image.size
    crop_width = min(size[0], width)
    crop_height = min(size[1], height)
    x1 = max(0, min(center[0] - crop_width // 2, width - crop_width))
    y1 = max(0, min(center[1] - crop_height // 2, height - crop_height))
    box: Box = (x1, y1, x1 + crop_width, y1 + crop_height)
    return image.crop(box), box


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


def draw_click_grid(
    image: Image.Image,
    cells: Iterable[GridCell],
    *,
    output_path: Path,
    selected_cell_id: str | None = None,
    gutter: int = 28,
) -> Path:
    source = image.convert("RGB")
    annotated = Image.new("RGB", (source.width + gutter, source.height + gutter), (255, 255, 255))
    annotated.paste(source, (gutter, gutter))
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    cells_list = list(cells)
    rows = max(cell.row for cell in cells_list)
    cols = max(cell.col for cell in cells_list)

    for col in range(1, cols + 1):
        matching = [cell for cell in cells_list if cell.col == col]
        x1 = min(cell.box[0] for cell in matching) + gutter
        x2 = max(cell.box[2] for cell in matching) + gutter
        label = str(col)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        draw.text(((x1 + x2 - (right - left)) // 2, (gutter - (bottom - top)) // 2), label, fill=(0, 0, 0), font=font)

    for row in range(1, rows + 1):
        matching = [cell for cell in cells_list if cell.row == row]
        y1 = min(cell.box[1] for cell in matching) + gutter
        y2 = max(cell.box[3] for cell in matching) + gutter
        label = str(row)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        draw.text(((gutter - (right - left)) // 2, (y1 + y2 - (bottom - top)) // 2), label, fill=(0, 0, 0), font=font)

    selected = cell_by_id(cells_list, selected_cell_id) if selected_cell_id else None
    line_color = (255, 0, 0)
    selected_color = (255, 210, 0)
    for cell in cells_list:
        box = (cell.box[0] + gutter, cell.box[1] + gutter, cell.box[2] + gutter, cell.box[3] + gutter)
        if selected and cell.id == selected.id:
            draw.rectangle(box, outline=selected_color, width=3)
        else:
            draw.rectangle(box, outline=line_color, width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    return output_path


def draw_full_click_marker(
    image: Image.Image,
    *,
    point: tuple[int, int],
    output_path: Path,
    label: str = "click_point",
) -> Path:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    x, y = point
    color = (255, 0, 0)

    draw.line((x - 14, y, x + 14, y), fill=color, width=5)
    draw.line((x, y - 14, x, y + 14), fill=color, width=5)
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=4)

    scale = 2
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_width = right - left
    text_height = bottom - top
    label_image = Image.new("RGBA", (text_width + 6, text_height + 4), (255, 255, 255, 230))
    label_draw = ImageDraw.Draw(label_image)
    label_draw.text((3, 2), label, fill=color, font=font)
    label_image = label_image.resize((label_image.width * scale, label_image.height * scale))
    label_x = min(max(0, x + 18), max(0, annotated.width - label_image.width))
    if y - label_image.height - 18 >= 0:
        label_y = y - label_image.height - 18
    else:
        label_y = min(y + 18, max(0, annotated.height - label_image.height))
    annotated.paste(label_image.convert("RGB"), (label_x, label_y))

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
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    draw.rectangle(box, outline=color, width=3)
    if label:
        left, top, right, bottom = draw.textbbox((box[0] + 4, max(0, box[1] - 14)), label, font=font)
        draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(255, 255, 255))
        draw.text((box[0] + 4, max(0, box[1] - 14)), label, fill=color, font=font)
    return annotated
