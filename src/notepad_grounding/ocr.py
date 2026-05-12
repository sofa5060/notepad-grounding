from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Iterable

from PIL import Image

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class OcrWord:
    text: str
    confidence: float
    box: Box
    line_id: int


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class OcrTile:
    index: int
    row: int
    col: int
    box: Box


class OcrError(RuntimeError):
    """Raised when OCR cannot run in the current environment."""


def extract_windows_ocr_lines(
    image: Image.Image,
    *,
    mode: str = "grid",
    scale: int = 2,
    tile_width: int = 320,
    tile_height: int = 240,
    overlap: int = 48,
) -> list[OcrLine]:
    if mode == "full":
        return extract_ocr_lines_from_full_image(image, ocr_words=extract_windows_ocr_words, scale=scale)
    if mode != "grid":
        raise OcrError(f"Unsupported OCR mode: {mode}")
    return extract_ocr_lines_from_grid(
        image,
        ocr_words=extract_windows_ocr_words,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
        scale=scale,
    )


def extract_ocr_lines_from_full_image(
    image: Image.Image,
    *,
    ocr_words: Callable[[Image.Image], list[OcrWord]],
    scale: int = 2,
) -> list[OcrLine]:
    ocr_image = prepare_ocr_image(image, scale=scale)
    words = ocr_words(ocr_image)
    lines = group_words_by_line(words)
    return scale_ocr_lines(lines, scale=scale)


def extract_ocr_lines_from_grid(
    image: Image.Image,
    *,
    ocr_words: Callable[[Image.Image], list[OcrWord]],
    tile_width: int = 320,
    tile_height: int = 240,
    overlap: int = 48,
    scale: int = 2,
) -> list[OcrLine]:
    words: list[OcrWord] = []
    for tile in iter_ocr_tiles(
        width=image.width,
        height=image.height,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
    ):
        crop = image.crop(tile.box)
        ocr_image = prepare_ocr_image(crop, scale=scale)
        tile_words = ocr_words(ocr_image)
        words.extend(
            offset_and_scale_words(
                tile_words,
                offset_x=tile.box[0],
                offset_y=tile.box[1],
                scale=scale,
                line_id_offset=tile.index * 1000,
            )
        )

    return group_words_by_line(dedupe_ocr_words(words))


def extract_windows_ocr_words(image: Image.Image) -> list[OcrWord]:
    try:
        return asyncio.run(_extract_windows_ocr_words_async(image))
    except RuntimeError as exc:
        if "asyncio.run() cannot be called" not in str(exc):
            raise

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_extract_windows_ocr_words_async(image))
    finally:
        loop.close()


async def _extract_windows_ocr_words_async(image: Image.Image) -> list[OcrWord]:
    try:
        from winrt.windows.graphics.imaging import BitmapDecoder
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.storage import FileAccessMode, StorageFile
    except ImportError as exc:
        raise OcrError("Windows OCR requires PyWinRT packages and must run on Windows.") from exc

    with NamedTemporaryFile(suffix=".png", delete=False) as file:
        temp_path = Path(file.name)

    try:
        image.convert("RGB").save(temp_path)
        storage_file = await StorageFile.get_file_from_path_async(str(temp_path))
        stream = await storage_file.open_async(FileAccessMode.READ)
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        engine = OcrEngine.try_create_from_user_profile_languages()
        if engine is None:
            raise OcrError("Windows OCR could not create an OCR engine for user languages.")
        result = await engine.recognize_async(bitmap)
        return _windows_result_to_words(result)
    except OcrError:
        raise
    except Exception as exc:
        raise OcrError(f"Windows OCR failed: {exc}") from exc
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _windows_result_to_words(result: Any) -> list[OcrWord]:
    words: list[OcrWord] = []
    for line_index, line in enumerate(result.lines, start=1):
        for word in line.words:
            text = str(word.text).strip()
            if not text:
                continue
            rect = word.bounding_rect
            box = (
                round(rect.x),
                round(rect.y),
                round(rect.x + rect.width),
                round(rect.y + rect.height),
            )
            words.append(OcrWord(text=text, confidence=100, box=box, line_id=line_index))
    return words


def prepare_ocr_image(image: Image.Image, *, scale: int = 2) -> Image.Image:
    if scale <= 1:
        return image.convert("RGB").copy()
    return image.convert("RGB").resize(
        (image.width * scale, image.height * scale),
        Image.Resampling.LANCZOS,
    )


def scale_ocr_lines(lines: Iterable[OcrLine], *, scale: int) -> list[OcrLine]:
    if scale <= 1:
        return list(lines)
    return [
        OcrLine(
            text=line.text,
            confidence=line.confidence,
            box=(
                round(line.box[0] / scale),
                round(line.box[1] / scale),
                round(line.box[2] / scale),
                round(line.box[3] / scale),
            ),
        )
        for line in lines
    ]


def iter_ocr_tiles(
    *,
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    overlap: int,
) -> list[OcrTile]:
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile width and height must be positive")
    if overlap < 0 or overlap >= min(tile_width, tile_height):
        raise ValueError("overlap must be non-negative and smaller than tile dimensions")

    stride_x = tile_width - overlap
    stride_y = tile_height - overlap
    x_starts = _tile_starts(length=width, tile_size=tile_width, stride=stride_x)
    y_starts = _tile_starts(length=height, tile_size=tile_height, stride=stride_y)

    tiles: list[OcrTile] = []
    index = 1
    for row, y in enumerate(y_starts):
        for col, x in enumerate(x_starts):
            tiles.append(
                OcrTile(
                    index=index,
                    row=row,
                    col=col,
                    box=(x, y, min(x + tile_width, width), min(y + tile_height, height)),
                )
            )
            index += 1
    return tiles


def offset_and_scale_words(
    words: Iterable[OcrWord],
    *,
    offset_x: int,
    offset_y: int,
    scale: int,
    line_id_offset: int,
) -> list[OcrWord]:
    return [
        OcrWord(
            text=word.text,
            confidence=word.confidence,
            box=(
                offset_x + round(word.box[0] / scale),
                offset_y + round(word.box[1] / scale),
                offset_x + round(word.box[2] / scale),
                offset_y + round(word.box[3] / scale),
            ),
            line_id=word.line_id + line_id_offset,
        )
        for word in words
    ]


def dedupe_ocr_words(words: Iterable[OcrWord], *, iou_threshold: float = 0.5) -> list[OcrWord]:
    kept: list[OcrWord] = []
    for word in sorted(words, key=lambda item: item.confidence, reverse=True):
        if any(_is_duplicate_word(word, existing, iou_threshold=iou_threshold) for existing in kept):
            continue
        kept.append(word)
    return sorted(kept, key=lambda item: (*_reading_order_key(item.box), item.text.lower()))


def group_words_by_line(
    words: Iterable[OcrWord],
    *,
    max_horizontal_gap: int = 32,
    max_vertical_gap: int = 8,
    max_wrapped_horizontal_gap: int = 24,
) -> list[OcrLine]:
    grouped: dict[int, list[OcrWord]] = defaultdict(list)
    for word in words:
        grouped[word.line_id].append(word)

    lines: list[OcrLine] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda word: _reading_order_key(word.box))
        clusters: list[list[OcrWord]] = []
        for word in ordered:
            if not clusters or word.box[0] - clusters[-1][-1].box[2] > max_horizontal_gap:
                clusters.append([word])
            else:
                clusters[-1].append(word)

        for cluster in clusters:
            text = " ".join(word.text for word in cluster)
            confidence = sum(word.confidence for word in cluster) / len(cluster)
            lines.append(OcrLine(text=text, confidence=confidence, box=_union(word.box for word in cluster)))

    return _merge_wrapped_lines(
        _merge_inline_neighbor_lines(
            sorted(lines, key=lambda line: _reading_order_key(line.box)),
            max_horizontal_gap=max_horizontal_gap,
        ),
        max_vertical_gap=max_vertical_gap,
        max_wrapped_horizontal_gap=max_wrapped_horizontal_gap,
    )


def _merge_inline_neighbor_lines(
    lines: list[OcrLine],
    *,
    max_horizontal_gap: int,
) -> list[OcrLine]:
    merged: list[OcrLine] = []
    for line in lines:
        if not merged:
            merged.append(line)
            continue

        previous = merged[-1]
        same_row = _vertical_overlap_ratio(previous.box, line.box) >= 0.55
        close_horizontal = _horizontal_gap(previous.box, line.box) <= max_horizontal_gap
        if same_row and close_horizontal:
            merged[-1] = _merge_lines(previous, line)
        else:
            merged.append(line)
    return merged


def _merge_wrapped_lines(
    lines: list[OcrLine],
    *,
    max_vertical_gap: int,
    max_wrapped_horizontal_gap: int,
) -> list[OcrLine]:
    merged: list[OcrLine] = []
    for line in lines:
        if not merged:
            merged.append(line)
            continue

        previous = merged[-1]
        vertical_gap = line.box[1] - previous.box[3]
        horizontal_gap = _horizontal_gap(previous.box, line.box)
        if 0 <= vertical_gap <= max_vertical_gap and horizontal_gap <= max_wrapped_horizontal_gap:
            merged[-1] = _merge_lines(previous, line)
        else:
            merged.append(line)
    return merged


def _merge_lines(first: OcrLine, second: OcrLine) -> OcrLine:
    return OcrLine(
        text=f"{first.text} {second.text}",
        confidence=(first.confidence + second.confidence) / 2,
        box=_union([first.box, second.box]),
    )


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


def _is_duplicate_word(candidate: OcrWord, existing: OcrWord, *, iou_threshold: float) -> bool:
    return _normalize(candidate.text) == _normalize(existing.text) and _box_iou(candidate.box, existing.box) >= iou_threshold


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
