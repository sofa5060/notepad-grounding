from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable

from notepad_grounding.grounding.annotations import Box
from notepad_grounding.grounding.ocr import OcrLine


@dataclass(frozen=True)
class IconCandidate:
    candidate_id: int
    label_text: str
    label_box: Box
    icon_box: Box
    combined_box: Box
    confidence_notes: list[str]

    @property
    def label_center(self) -> tuple[float, float]:
        return _center(self.label_box)

    @property
    def icon_center(self) -> tuple[float, float]:
        return _center(self.icon_box)


def infer_icon_candidates(
    labels: Iterable[OcrLine],
    *,
    screen_size: tuple[int, int],
    min_icon_size: int = 32,
    taskbar_height: int = 48,
) -> list[IconCandidate]:
    label_list = list(labels)
    usable_labels = [
        label
        for label in label_list
        if _is_plausible_desktop_label(label)
        and not _is_in_taskbar(label.box, screen_size=screen_size, taskbar_height=taskbar_height)
    ]
    if not usable_labels:
        return []

    median_text_height = int(median(max(1, _height(label.box)) for label in usable_labels))
    fallback_icon_size = max(48, 3 * median_text_height)
    max_icon_size = min(160, 8 * median_text_height)
    desktop_cell_width = _estimate_desktop_cell_width(usable_labels)

    candidates: list[IconCandidate] = []
    for label in usable_labels:
        candidate_id = len(candidates) + 1
        icon_box = _infer_icon_box(
            label.box,
            screen_size=screen_size,
            min_icon_size=min_icon_size,
            fallback_icon_size=fallback_icon_size,
            max_icon_size=max_icon_size,
            desktop_cell_width=desktop_cell_width,
        )
        combined_box = _union_boxes([icon_box, label.box], label=f"#{candidate_id} {label.text}")
        notes = [
            f"median_text_height={median_text_height}",
            f"min_icon_size={min_icon_size}",
            f"fallback_icon_size={fallback_icon_size}",
            f"max_icon_size={max_icon_size}",
        ]
        if desktop_cell_width:
            notes.append(f"desktop_cell_width={desktop_cell_width}")

        candidates.append(
            IconCandidate(
                candidate_id=candidate_id,
                label_text=label.text,
                label_box=label.box,
                icon_box=icon_box,
                combined_box=combined_box,
                confidence_notes=notes,
            )
        )

    return candidates


def _infer_icon_box(
    label_box: Box,
    *,
    screen_size: tuple[int, int],
    min_icon_size: int,
    fallback_icon_size: int,
    max_icon_size: int,
    desktop_cell_width: int | None,
) -> Box:
    label_width = _width(label_box)
    label_height = _height(label_box)
    icon_size = max(min_icon_size, fallback_icon_size, label_height * 3)
    icon_size = min(icon_size, max_icon_size)

    search_width = max(label_width + 32, 96)
    if desktop_cell_width:
        search_width = min(search_width, desktop_cell_width)
    icon_size = min(icon_size, search_width, max_icon_size)
    icon_size = max(icon_size, min_icon_size)

    label_center_x, _ = _center(label_box)
    gap = max(4, min(12, label_height // 2))
    x1 = round(label_center_x - icon_size / 2)
    x2 = x1 + icon_size
    y2 = label_box.y1 - gap
    y1 = y2 - icon_size

    return _clamp_box(Box(x1, y1, x2, y2, "icon"), screen_size)


def _estimate_desktop_cell_width(labels: list[OcrLine]) -> int | None:
    centers = sorted(_center(label.box)[0] for label in labels)
    gaps = [
        round(right - left)
        for left, right in zip(centers, centers[1:])
        if 40 <= right - left <= 240
    ]
    if not gaps:
        return None
    return max(96, int(median(gaps)))


def _is_in_taskbar(
    box: Box,
    *,
    screen_size: tuple[int, int],
    taskbar_height: int,
) -> bool:
    _, screen_height = screen_size
    taskbar_top = screen_height - taskbar_height
    return box.y1 >= taskbar_top


def _is_plausible_desktop_label(label: OcrLine) -> bool:
    text = label.text.strip()
    if not any(character.isalnum() for character in text):
        return False
    if _width(label.box) < 6 or _height(label.box) < 6:
        return False
    if _width(label.box) > 420 or _height(label.box) > 96:
        return False
    return True


def _union_boxes(boxes: list[Box], *, label: str) -> Box:
    return Box(
        min(box.x1 for box in boxes),
        min(box.y1 for box in boxes),
        max(box.x2 for box in boxes),
        max(box.y2 for box in boxes),
        label,
    )


def _clamp_box(box: Box, screen_size: tuple[int, int]) -> Box:
    width, height = screen_size
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    x1 = min(max(box.x1, 0), max_x)
    y1 = min(max(box.y1, 0), max_y)
    x2 = min(max(box.x2, x1), max_x)
    y2 = min(max(box.y2, y1), max_y)
    return Box(x1, y1, x2, y2, box.label)


def _center(box: Box) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)


def _width(box: Box) -> int:
    return box.x2 - box.x1


def _height(box: Box) -> int:
    return box.y2 - box.y1
