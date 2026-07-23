from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from grid_ocr.ocr import Box
from grid_ocr.ocr import OcrLine
from grid_ocr.ocr import _normalize
from grid_ocr.ocr import _union

TASKBAR_HEIGHT = 48
ICON_SIZE = 64
LABEL_TO_ICON_GAP = 6
MIN_SCORE = 0.72


@dataclass(frozen=True)
class Candidate:
    label_text: str
    label_box: Box
    icon_box: Box
    score: float

    @property
    def icon_center(self) -> tuple[int, int]:
        (left, top, right, bottom) = self.icon_box
        return ((left + right) // 2, (top + bottom) // 2)

    @property
    def combined_box(self) -> Box:
        return _union([self.label_box, self.icon_box])


def locate_from_lines(lines: Iterable[OcrLine], *, screen_size: tuple[int, int], query: str) -> Candidate | None:
    candidates = find_candidates(lines, screen_size=screen_size, query=query)
    if not candidates:
        return None

    best = max(candidates, key=lambda candidate: candidate.score)
    if best.score < MIN_SCORE:
        return None
    return best


def find_candidates(lines: Iterable[OcrLine], *, screen_size: tuple[int, int], query: str) -> list[Candidate]:
    (width, height) = screen_size
    candidates: list[Candidate] = []
    for line in lines:
        (left, top, right, bottom) = line.box

        # 1. Keep only plausible icon labels: contains letters/digits, label-sized, above the taskbar.
        if not any(character.isalnum() for character in line.text):
            continue
        if not (4 <= right - left <= 420) or not (4 <= bottom - top <= 96):
            continue
        if top >= height - TASKBAR_HEIGHT:
            continue

        # 2. Score the label against the query: 1.0 on exact normalized match, else fuzzy ratio.
        normalized_query = _normalize(query)
        normalized_text = _normalize(line.text)
        if normalized_query == normalized_text:
            score = 1.0
        else:
            score = SequenceMatcher(None, normalized_query, normalized_text).ratio()

        # 3. Assume the icon sits centered directly above its label: a 64px box, 6px up, clamped to screen.
        icon_left = (left + right) // 2 - ICON_SIZE // 2
        icon_bottom = top - LABEL_TO_ICON_GAP
        icon_box = (
            max(0, min(icon_left, width - 1)),
            max(0, min(icon_bottom - ICON_SIZE, height - 1)),
            max(0, min(icon_left + ICON_SIZE, width - 1)),
            max(0, min(icon_bottom, height - 1)),
        )

        candidates.append(Candidate(label_text=line.text, label_box=line.box, icon_box=icon_box, score=score))
    return candidates
