from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from notepad_grounding.shared.geometry import Box


@dataclass(frozen=True)
class ClickPoint:
    id: str
    center: tuple[int, int]


def build_click_points(
    image_size: tuple[int, int],
    *,
    rows: int,
    cols: int,
    margin: int = 24,
) -> list[ClickPoint]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    width, height = image_size
    max_margin_x = max(0, (width - 1) // 2)
    max_margin_y = max(0, (height - 1) // 2)
    margin_x = min(margin, max_margin_x)
    margin_y = min(margin, max_margin_y)
    usable_width = max(0, width - (2 * margin_x))
    usable_height = max(0, height - (2 * margin_y))
    points: list[ClickPoint] = []

    for row in range(rows):
        for col in range(cols):
            index = row * cols + col + 1
            x = margin_x + round(usable_width * col / max(1, cols - 1))
            y = margin_y + round(usable_height * row / max(1, rows - 1))
            points.append(ClickPoint(id=f"P{index:02d}", center=(x, y)))
    return points


def point_by_id(points: Iterable[ClickPoint], point_id: str) -> ClickPoint:
    for point in points:
        if point.id == point_id:
            return point
    raise ValueError(f"Unknown click point: {point_id}")


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


def offset_point(point: tuple[int, int], *, offset: tuple[int, int]) -> tuple[int, int]:
    return (point[0] + offset[0], point[1] + offset[1])


def draw_click_points(
    image: Image.Image,
    points: Iterable[ClickPoint],
    *,
    output_path: Path,
    selected_point_id: str | None = None,
) -> Path:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for point in points:
        x, y = point.center
        color = (255, 0, 0) if point.id == selected_point_id else (220, 0, 0)
        width = 3 if point.id == selected_point_id else 2
        draw.line((x - 6, y, x + 6, y), fill=color, width=width)
        draw.line((x, y - 6, x, y + 6), fill=color, width=width)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=width)

        label_x = min(max(0, x + 8), max(0, annotated.width - 24))
        label_y = min(max(0, y - 10), max(0, annotated.height - 12))
        left, top, right, bottom = draw.textbbox((label_x, label_y), point.id, font=font)
        draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(255, 255, 255))
        draw.text((label_x, label_y), point.id, fill=color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    return output_path
