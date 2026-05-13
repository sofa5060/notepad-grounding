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
) -> Path:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for cell in cells:
        color = (255, 0, 0) if cell.id == selected_cell_id else (255, 210, 0)
        width = 4 if cell.id == selected_cell_id else 2
        draw.rectangle(cell.box, outline=color, width=width)
        label_x = cell.box[0] + 4
        label_y = cell.box[1] + 4
        left, top, right, bottom = draw.textbbox((label_x, label_y), cell.id, font=font)
        draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(255, 255, 255))
        draw.text((label_x, label_y), cell.id, fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    return output_path
