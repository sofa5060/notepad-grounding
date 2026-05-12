from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable

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


class OcrError(RuntimeError):
    """Raised when OCR cannot run in the current environment."""


def extract_windows_ocr_lines(image: Image.Image) -> list[OcrLine]:
    words = extract_windows_ocr_words(image)
    return group_words_by_line(words)


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


def group_words_by_line(words: Iterable[OcrWord], *, max_gap: int = 32) -> list[OcrLine]:
    grouped: dict[int, list[OcrWord]] = defaultdict(list)
    for word in words:
        grouped[word.line_id].append(word)

    lines: list[OcrLine] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda word: (word.box[1], word.box[0]))
        clusters: list[list[OcrWord]] = []
        for word in ordered:
            if not clusters or word.box[0] - clusters[-1][-1].box[2] > max_gap:
                clusters.append([word])
            else:
                clusters[-1].append(word)

        for cluster in clusters:
            text = " ".join(word.text for word in cluster)
            confidence = sum(word.confidence for word in cluster) / len(cluster)
            lines.append(OcrLine(text=text, confidence=confidence, box=_union(word.box for word in cluster)))

    return sorted(lines, key=lambda line: (line.box[1], line.box[0]))


def _union(boxes: Iterable[Box]) -> Box:
    box_list = list(boxes)
    return (
        min(box[0] for box in box_list),
        min(box[1] for box in box_list),
        max(box[2] for box in box_list),
        max(box[3] for box in box_list),
    )
