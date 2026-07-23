from __future__ import annotations

import asyncio
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

from PIL import Image
from winrt.windows.graphics.imaging import BitmapDecoder
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage import FileAccessMode, StorageFile

Box = tuple[int, int, int, int]

TILE_WIDTH = 320
TILE_HEIGHT = 240
TILE_OVERLAP = 48
OCR_SCALE = 2
IOU_DUPLICATE_THRESHOLD = 0.5
MAX_HORIZONTAL_GAP = 32
MAX_WRAPPED_VERTICAL_GAP = 8
MAX_WRAPPED_HORIZONTAL_GAP = 24


@dataclass(frozen=True)
class OcrWord:
    text: str
    confidence: float
    box: Box
    line_id: Hashable


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    box: Box


class OcrError(RuntimeError):
    """Raised when OCR cannot run in the current environment."""


async def extract_windows_ocr_words(image: Image.Image) -> list[OcrWord]:
    # Windows locks open files, so close the handle before PIL and WinRT open the
    # path themselves; delete_on_close=False keeps the file until the with-block exits.
    with NamedTemporaryFile(suffix=".png", delete_on_close=False) as file:
        file.close()
        temp_path = Path(file.name)
        image.save(temp_path)
        storage_file = await StorageFile.get_file_from_path_async(str(temp_path))
        stream = await storage_file.open_async(FileAccessMode.READ)
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        stream.close()
        engine = OcrEngine.try_create_from_user_profile_languages()
        if engine is None:
            raise OcrError("Windows OCR could not create an OCR engine for user languages.")
        result = await engine.recognize_async(bitmap)

        words: list[OcrWord] = []
        for line_index, line in enumerate(result.lines, start=1):
            for word in line.words:
                text = str(word.text).strip()
                if not text:
                    continue
                rect = word.bounding_rect
                box = (round(rect.x), round(rect.y), round(rect.x + rect.width), round(rect.y + rect.height))
                words.append(OcrWord(text=text, confidence=100, box=box, line_id=line_index))
        return words


def extract_ocr_lines_from_grid(
    image: Image.Image,
    *,
    tile_width: int = TILE_WIDTH,
    tile_height: int = TILE_HEIGHT,
    overlap: int = TILE_OVERLAP,
    scale: int = OCR_SCALE,
) -> list[OcrLine]:
    # Upscale the whole screenshot once; each tile is cropped from the scaled image.
    scaled = image.convert("RGB")
    if scale > 1:
        scaled = scaled.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)

    x_starts = _tile_starts(length=image.width, tile_size=tile_width, stride=tile_width - overlap)
    y_starts = _tile_starts(length=image.height, tile_size=tile_height, stride=tile_height - overlap)
    tiles = [(x, y) for y in y_starts for x in x_starts]

    words: list[OcrWord] = []
    for tile_index, (x, y) in enumerate(tiles, start=1):
        x2 = min(x + tile_width, image.width)
        y2 = min(y + tile_height, image.height)
        crop = scaled.crop((x * scale, y * scale, x2 * scale, y2 * scale))
        for word in asyncio.run(extract_windows_ocr_words(crop)):
            words.append(
                OcrWord(
                    text=word.text,
                    confidence=word.confidence,
                    box=(
                        x + round(word.box[0] / scale),
                        y + round(word.box[1] / scale),
                        x + round(word.box[2] / scale),
                        y + round(word.box[3] / scale),
                    ),
                    line_id=(tile_index, word.line_id),
                )
            )
    return group_words_by_line(dedupe_ocr_words(words))


def dedupe_ocr_words(words: Iterable[OcrWord]) -> list[OcrWord]:
    kept: list[OcrWord] = []
    for word in sorted(words, key=lambda item: item.confidence, reverse=True):
        is_duplicate = any(
            _normalize(existing.text) == _normalize(word.text)
            and _box_iou(word.box, existing.box) >= IOU_DUPLICATE_THRESHOLD
            for existing in kept
        )
        if not is_duplicate:
            kept.append(word)
    return kept


def group_words_by_line(words: Iterable[OcrWord], *, max_horizontal_gap: int = MAX_HORIZONTAL_GAP) -> list[OcrLine]:
    # 1. Bucket words by the OCR line they came from (per tile).
    grouped: dict[Hashable, list[OcrWord]] = {}
    for word in words:
        grouped.setdefault(word.line_id, []).append(word)

    # 2. Within each bucket, split at large horizontal gaps; each cluster becomes one line.
    lines: list[OcrLine] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda word: _reading_order_key(word.box))
        clusters: list[list[OcrWord]] = []
        for word in ordered:
            if not clusters:
                clusters.append([word])
                continue
            previous_word = clusters[-1][-1]
            gap = word.box[0] - previous_word.box[2]
            if gap > max_horizontal_gap:
                clusters.append([word])
            else:
                clusters[-1].append(word)

        for cluster in clusters:
            text = " ".join(word.text for word in cluster)
            confidence = sum(word.confidence for word in cluster) / len(cluster)
            lines.append(OcrLine(text=text, confidence=confidence, box=_union(word.box for word in cluster)))
    lines.sort(key=lambda line: _reading_order_key(line.box))

    # 3. Merge fragments sitting on the same row — one label read half-and-half by two tiles.
    inline_merged: list[OcrLine] = []
    for line in lines:
        previous = inline_merged[-1] if inline_merged else None
        same_row = previous is not None and _vertical_overlap_ratio(previous.box, line.box) >= 0.55
        if same_row and _horizontal_gap(previous.box, line.box) <= max_horizontal_gap:
            inline_merged[-1] = OcrLine(
                text=f"{previous.text} {line.text}",
                confidence=(previous.confidence + line.confidence) / 2,
                box=_union([previous.box, line.box]),
            )
        else:
            inline_merged.append(line)

    # 4. Merge a line into the one directly above it — a label wrapped onto two rows.
    merged: list[OcrLine] = []
    for line in inline_merged:
        previous = merged[-1] if merged else None
        wraps = previous is not None and 0 <= line.box[1] - previous.box[3] <= MAX_WRAPPED_VERTICAL_GAP
        if wraps and _horizontal_gap(previous.box, line.box) <= MAX_WRAPPED_HORIZONTAL_GAP:
            merged[-1] = OcrLine(
                text=f"{previous.text} {line.text}",
                confidence=(previous.confidence + line.confidence) / 2,
                box=_union([previous.box, line.box]),
            )
        else:
            merged.append(line)
    return merged


def _union(boxes: Iterable[Box]) -> Box:
    box_list = list(boxes)
    return (
        min(box[0] for box in box_list),
        min(box[1] for box in box_list),
        max(box[2] for box in box_list),
        max(box[3] for box in box_list),
    )


def _horizontal_gap(first: Box, second: Box) -> int:
    if first[2] < second[0]:
        return second[0] - first[2]
    if second[2] < first[0]:
        return first[0] - second[2]
    return 0


def _vertical_overlap_ratio(first: Box, second: Box) -> float:
    top = max(first[1], second[1])
    bottom = min(first[3], second[3])
    overlap = max(0, bottom - top)
    shortest = max(1, min(first[3] - first[1], second[3] - second[1]))
    return overlap / shortest


def _box_iou(first: Box, second: Box) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _normalize(text: str) -> str:
    return "".join(character.lower() for character in text if character.isalnum())


def _tile_starts(*, length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _reading_order_key(box: Box) -> tuple[int, int]:
    return (box[1] // 12, box[0])
