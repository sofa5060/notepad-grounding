from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from grid_ocr.ocr import Box
from grid_ocr.ocr import OcrLine


@dataclass(frozen=True)
class GridCell:
    index: int
    row: int
    col: int
    box: Box


@dataclass(frozen=True)
class Candidate:
    id: int
    label_text: str
    label_box: Box
    icon_box: Box
    combined_box: Box
    score: float
    score_notes: list[str]

    @property
    def icon_center(self) -> tuple[int, int]:
        return _center(self.icon_box)


@dataclass(frozen=True)
class LocateResult:
    query: str
    center: tuple[int, int]
    candidate: Candidate
    candidates: list[Candidate]
    grid: list[GridCell]


def locate_from_lines(
    lines: Iterable[OcrLine],
    *,
    screen_size: tuple[int, int],
    query: str,
    cell_width: int = 96,
    cell_height: int = 96,
    taskbar_height: int = 48,
    min_score: float = 0.72,
) -> LocateResult | None:
    grid = build_grid(
        screen_size,
        cell_width=cell_width,
        cell_height=cell_height,
        taskbar_height=taskbar_height,
    )
    candidates = infer_candidates(
        lines,
        screen_size=screen_size,
        query=query,
        taskbar_height=taskbar_height,
    )
    if not candidates:
        return None

    best = max(candidates, key=lambda candidate: candidate.score)
    if best.score < min_score:
        return None
    return LocateResult(
        query=query,
        center=best.icon_center,
        candidate=best,
        candidates=candidates,
        grid=grid,
    )


def build_grid(
    screen_size: tuple[int, int],
    *,
    cell_width: int = 96,
    cell_height: int = 96,
    taskbar_height: int = 48,
) -> list[GridCell]:
    width, height = screen_size
    usable_height = max(0, height - taskbar_height)
    cells: list[GridCell] = []
    index = 1
    row = 0
    for y in range(0, usable_height, cell_height):
        col = 0
        y2 = min(y + cell_height, usable_height)
        for x in range(0, width, cell_width):
            x2 = min(x + cell_width, width)
            cells.append(GridCell(index=index, row=row, col=col, box=(x, y, x2, y2)))
            index += 1
            col += 1
        row += 1
    return cells


def infer_candidates(
    lines: Iterable[OcrLine],
    *,
    screen_size: tuple[int, int],
    query: str,
    taskbar_height: int = 48,
    icon_size: int = 64,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for line in lines:
        if not _is_plausible_label(line, screen_size=screen_size, taskbar_height=taskbar_height):
            continue

        score = label_score(query, line.text)
        icon_box = _infer_icon_box(line.box, screen_size=screen_size, icon_size=icon_size)
        candidate_id = len(candidates) + 1
        candidates.append(
            Candidate(
                id=candidate_id,
                label_text=line.text,
                label_box=line.box,
                icon_box=icon_box,
                combined_box=_union([line.box, icon_box]),
                score=score,
                score_notes=[f"label_score={score:.3f}"],
            )
        )
    return candidates


def label_score(query: str, text: str) -> float:
    normalized_query = _normalize(query)
    normalized_text = _normalize(text)
    if not normalized_query or not normalized_text:
        return 0.0
    if normalized_query == normalized_text:
        return 1.0
    return SequenceMatcher(None, normalized_query, normalized_text).ratio()


def _infer_icon_box(
    label_box: Box,
    *,
    screen_size: tuple[int, int],
    icon_size: int,
) -> Box:
    label_center_x, _ = _center(label_box)
    gap = 6
    x1 = label_center_x - icon_size // 2
    x2 = x1 + icon_size
    y2 = label_box[1] - gap
    y1 = y2 - icon_size
    return _clamp((x1, y1, x2, y2), screen_size)


def _is_plausible_label(
    line: OcrLine,
    *,
    screen_size: tuple[int, int],
    taskbar_height: int,
) -> bool:
    text = line.text.strip()
    if not any(character.isalnum() for character in text):
        return False
    width = line.box[2] - line.box[0]
    height = line.box[3] - line.box[1]
    if width < 4 or height < 4 or width > 420 or height > 96:
        return False
    return line.box[1] < screen_size[1] - taskbar_height


def _normalize(text: str) -> str:
    return "".join(character.lower() for character in text if character.isalnum())


def _center(box: Box) -> tuple[int, int]:
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def _union(boxes: list[Box]) -> Box:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _clamp(box: Box, screen_size: tuple[int, int]) -> Box:
    width, height = screen_size
    return (
        max(0, min(box[0], width - 1)),
        max(0, min(box[1], height - 1)),
        max(0, min(box[2], width - 1)),
        max(0, min(box[3], height - 1)),
    )
