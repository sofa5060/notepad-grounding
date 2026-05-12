from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable

from PIL import Image, ImageEnhance, ImageOps

from notepad_grounding.grounding.annotations import Box


@dataclass(frozen=True)
class OcrWord:
    text: str
    confidence: float
    box: Box
    block_num: int
    par_num: int
    line_num: int
    word_num: int


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    box: Box
    word_count: int = 1


class OcrError(RuntimeError):
    """Raised when OCR cannot be executed."""


def extract_ocr_lines(
    image: Image.Image,
    *,
    min_confidence: float = 50,
    tesseract_cmd: str | None = None,
    upscale_factor: int = 2,
    preprocess_mode: str = "desktop_label",
) -> list[OcrLine]:
    """Run Tesseract OCR and return grouped desktop-label-like text lines."""

    words = extract_ocr_words(
        image,
        min_confidence=min_confidence,
        tesseract_cmd=tesseract_cmd,
        upscale_factor=upscale_factor,
        preprocess_mode=preprocess_mode,
    )
    return group_words_by_line(words)


def extract_ocr_words(
    image: Image.Image,
    *,
    min_confidence: float = 50,
    tesseract_cmd: str | None = None,
    upscale_factor: int = 2,
    preprocess_mode: str = "desktop_label",
) -> list[OcrWord]:
    """Run Tesseract OCR and return raw word boxes in screenshot coordinates.

    Args:
        image: Input screenshot image.
        min_confidence: Minimum Tesseract confidence to keep a word.
        tesseract_cmd: Optional path to Tesseract executable.
        upscale_factor: Factor to upscale before OCR (higher = more accurate on small text).
        preprocess_mode: "desktop_label" for icon-label-specific preprocessing,
            "standard" for generic grayscale + contrast.
    """

    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as exc:
        raise OcrError("Missing OCR dependency 'pytesseract'. Run 'uv sync'.") from exc

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    if preprocess_mode == "desktop_label":
        ocr_image = prepare_desktop_label_image(image, upscale_factor=upscale_factor)
    else:
        ocr_image = prepare_image_for_ocr(image, upscale_factor=upscale_factor)

    try:
        data = pytesseract.image_to_data(
            ocr_image,
            output_type=Output.DICT,
            config="--psm 11",
        )
    except Exception as exc:
        raise OcrError(
            "Tesseract OCR failed. Install Tesseract in Windows or pass "
            f"--tesseract-cmd with the executable path. Details: {exc}"
        ) from exc

    return extract_words_from_tesseract_data(
        data,
        min_confidence=min_confidence,
        coordinate_scale=upscale_factor,
    )


def extract_ocr_words_tiled(
    image: Image.Image,
    *,
    min_confidence: float = 50,
    tesseract_cmd: str | None = None,
    upscale_factor: int = 2,
    tile_size: int = 360,
    overlap: int = 80,
    preprocess_mode: str = "desktop_label",
) -> list[OcrWord]:
    """Run OCR over overlapping crops and return deduplicated screen coordinates."""

    words: list[OcrWord] = []
    for tile_index, tile in enumerate(
        iter_ocr_tiles(width=image.width, height=image.height, tile_size=tile_size, overlap=overlap),
        start=1,
    ):
        crop = image.crop((tile.x1, tile.y1, tile.x2, tile.y2))
        tile_words = extract_ocr_words(
            crop,
            min_confidence=min_confidence,
            tesseract_cmd=tesseract_cmd,
            upscale_factor=upscale_factor,
            preprocess_mode=preprocess_mode,
        )
        words.extend(
            offset_ocr_words(
                tile_words,
                offset_x=tile.x1,
                offset_y=tile.y1,
                group_offset=tile_index * 1000,
            )
        )

    return dedupe_ocr_words(words)


def extract_windows_ocr_words(image: Image.Image) -> list[OcrWord]:
    """Run Windows.Media.Ocr over a Pillow image.

    This backend is only available on Windows with the PyWinRT packages installed.
    """

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


def extract_windows_ocr_words_tiled(
    image: Image.Image,
    *,
    tile_size: int = 400,
    overlap: int = 50,
) -> list[OcrWord]:
    """Run Windows OCR over overlapping crops for focused detection.

    Windows OCR works well on raw screenshots but may pick up UI text from
    open windows. Tiling into smaller crops helps localize detection and
    reduces false positives from large windowed applications.
    """

    words: list[OcrWord] = []
    for tile_index, tile in enumerate(
        iter_ocr_tiles(width=image.width, height=image.height, tile_size=tile_size, overlap=overlap),
        start=1,
    ):
        crop = image.crop((tile.x1, tile.y1, tile.x2, tile.y2))
        tile_words = extract_windows_ocr_words(crop)
        words.extend(
            offset_ocr_words(
                tile_words,
                offset_x=tile.x1,
                offset_y=tile.y1,
                group_offset=tile_index * 1000,
            )
        )

    return dedupe_ocr_words(words, iou_threshold=0.35)


async def _extract_windows_ocr_words_async(image: Image.Image) -> list[OcrWord]:
    try:
        from winrt.windows.graphics.imaging import BitmapDecoder
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.storage import FileAccessMode, StorageFile
    except ImportError as exc:
        raise OcrError(
            "Windows OCR backend requires PyWinRT packages. Run 'uv sync' on Windows."
        ) from exc

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
        return windows_ocr_result_to_words(result)
    except OcrError:
        raise
    except Exception as exc:
        raise OcrError(f"Windows OCR failed: {exc}") from exc
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def windows_ocr_result_to_words(result: Any) -> list[OcrWord]:
    words: list[OcrWord] = []
    for line_index, line in enumerate(result.lines, start=1):
        for word_index, word in enumerate(line.words, start=1):
            text = str(word.text).strip()
            if not text:
                continue
            rect = word.bounding_rect
            x1 = round(rect.x)
            y1 = round(rect.y)
            x2 = round(rect.x + rect.width)
            y2 = round(rect.y + rect.height)
            words.append(
                OcrWord(
                    text=text,
                    confidence=100,
                    box=Box(x1, y1, x2, y2, text),
                    block_num=line_index,
                    par_num=1,
                    line_num=line_index,
                    word_num=word_index,
                )
            )
    return words


def iter_ocr_tiles(
    *,
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
) -> list[Box]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be greater than or equal to 0 and less than tile_size")

    stride = tile_size - overlap
    x_starts = _tile_starts(length=width, tile_size=tile_size, stride=stride)
    y_starts = _tile_starts(length=height, tile_size=tile_size, stride=stride)
    return [
        Box(x, y, min(x + tile_size, width), min(y + tile_size, height), "tile")
        for y in y_starts
        for x in x_starts
    ]


def offset_ocr_words(
    words: Iterable[OcrWord],
    *,
    offset_x: int,
    offset_y: int,
    group_offset: int,
) -> list[OcrWord]:
    offset_words: list[OcrWord] = []
    for word in words:
        box = Box(
            word.box.x1 + offset_x,
            word.box.y1 + offset_y,
            word.box.x2 + offset_x,
            word.box.y2 + offset_y,
            word.box.label,
        )
        offset_words.append(
            OcrWord(
                text=word.text,
                confidence=word.confidence,
                box=box,
                block_num=word.block_num + group_offset,
                par_num=word.par_num,
                line_num=word.line_num,
                word_num=word.word_num,
            )
        )
    return offset_words


def dedupe_ocr_words(
    words: Iterable[OcrWord],
    *,
    iou_threshold: float = 0.45,
) -> list[OcrWord]:
    kept: list[OcrWord] = []
    for word in sorted(words, key=lambda item: item.confidence, reverse=True):
        if any(_is_duplicate_word(word, existing, iou_threshold=iou_threshold) for existing in kept):
            continue
        kept.append(word)
    return sorted(kept, key=lambda item: (item.box.y1, item.box.x1, item.text.lower()))


def prepare_image_for_ocr(image: Image.Image, *, upscale_factor: int = 2) -> Image.Image:
    """Improve desktop-label OCR by enlarging and increasing text contrast."""

    grayscale = ImageOps.grayscale(image)
    high_contrast = ImageEnhance.Contrast(grayscale).enhance(2.5)
    if upscale_factor <= 1:
        return high_contrast
    return high_contrast.resize(
        (image.width * upscale_factor, image.height * upscale_factor),
        Image.Resampling.LANCZOS,
    )


def prepare_desktop_label_image(image: Image.Image, *, upscale_factor: int = 3) -> Image.Image:
    """Preprocess screenshot specifically for desktop icon label OCR.

    Desktop icon labels are rendered as white text with a dark drop shadow
    on top of arbitrary wallpaper. Standard OCR preprocessing (grayscale +
    contrast) does not amplify this specific signature and often misses small
    labels.

    This function:
      1. Detects "white text" pixels (high, balanced RGB values)
      2. Detects "dark shadow" pixels (low luminance)
      3. Creates a text probability map that combines both signals
      4. Produces a high-contrast output where text is white (255),
         shadow is black (0), and wallpaper is suppressed to mid-gray
      5. Upscales for better OCR accuracy on small fonts

    The result looks like a synthetic document image, which OCR engines
    handle much more reliably than raw desktop screenshots.
    """

    rgb = image.convert("RGB")
    pixels = rgb.load()
    width, height = rgb.size

    # Pass 1: compute per-pixel text-likeness scores
    # White text: all channels high and similar
    # Dark shadow: all channels low
    text_score = [[0.0 for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            luminance = (0.299 * r + 0.587 * g + 0.114 * b)
            max_channel = max(r, g, b)
            min_channel = min(r, g, b)
            channel_spread = max_channel - min_channel

            # White text score: high luminance, low spread (balanced channels)
            whiteness = 0.0
            if luminance >= 160 and channel_spread <= 40:
                whiteness = (luminance - 160) / 95.0  # 160-255 -> 0.0-1.0

            # Dark shadow score: low luminance
            shadowness = 0.0
            if luminance <= 80:
                shadowness = (80 - luminance) / 80.0  # 0-80 -> 1.0-0.0

            text_score[y][x] = max(whiteness, shadowness * 0.6)

    # Pass 2: local enhancement — boost pixels that are text-like or
    # adjacent to text-like pixels (catches faint shadow edges)
    enhanced = Image.new("L", (width, height), 128)
    enhanced_pixels = enhanced.load()

    for y in range(height):
        for x in range(width):
            score = text_score[y][x]

            # Also consider neighborhood: a pixel near strong text is likely text
            neighborhood_max = score
            if score < 0.5:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            neighborhood_max = max(neighborhood_max, text_score[ny][nx])

            combined = max(score, neighborhood_max * 0.7)

            # Map to output: text -> white, background -> mid-gray
            if combined >= 0.35:
                value = int(128 + combined * 127)
            else:
                value = int(128 - (0.35 - combined) * 200)
                value = max(0, value)

            enhanced_pixels[x, y] = min(255, value)

    # Pass 3: global contrast boost
    high_contrast = ImageEnhance.Contrast(enhanced).enhance(3.0)

    # Pass 4: upscale
    if upscale_factor > 1:
        return high_contrast.resize(
            (width * upscale_factor, height * upscale_factor),
            Image.Resampling.LANCZOS,
        )
    return high_contrast


def _tile_starts(*, length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _is_duplicate_word(
    candidate: OcrWord,
    existing: OcrWord,
    *,
    iou_threshold: float,
) -> bool:
    if _normalize_text(candidate.text) != _normalize_text(existing.text):
        return False
    return _box_iou(candidate.box, existing.box) >= iou_threshold


def _normalize_text(text: str) -> str:
    return "".join(character.lower() for character in text if character.isalnum())


def _box_iou(first: Box, second: Box) -> float:
    inter_x1 = max(first.x1, second.x1)
    inter_y1 = max(first.y1, second.y1)
    inter_x2 = min(first.x2, second.x2)
    inter_y2 = min(first.y2, second.y2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    if intersection == 0:
        return 0

    first_area = max(0, first.x2 - first.x1) * max(0, first.y2 - first.y1)
    second_area = max(0, second.x2 - second.x1) * max(0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    if union == 0:
        return 0
    return intersection / union


def extract_words_from_tesseract_data(
    data: dict[str, list[Any]],
    *,
    min_confidence: float = 50,
    coordinate_scale: int = 1,
) -> list[OcrWord]:
    words: list[OcrWord] = []
    for index, raw_text in enumerate(data.get("text", [])):
        text = str(raw_text).strip()
        if not text:
            continue

        confidence = _parse_confidence(data["conf"][index])
        if confidence < min_confidence:
            continue

        left = _scale_coordinate(data["left"][index], coordinate_scale)
        top = _scale_coordinate(data["top"][index], coordinate_scale)
        width = _scale_coordinate(data["width"][index], coordinate_scale)
        height = _scale_coordinate(data["height"][index], coordinate_scale)
        box = Box(left, top, left + width, top + height, text)

        words.append(
            OcrWord(
                text=text,
                confidence=confidence,
                box=box,
                block_num=int(data.get("block_num", [0])[index]),
                par_num=int(data.get("par_num", [0])[index]),
                line_num=int(data.get("line_num", [0])[index]),
                word_num=int(data.get("word_num", [index])[index]),
            )
        )

    return words


def group_words_by_line(
    words: Iterable[OcrWord],
    *,
    max_vertical_gap: int = 6,
    max_center_delta: int = 48,
    max_horizontal_gap: int = 48,
) -> list[OcrLine]:
    """Group OCR words into desktop label groups.

    Tesseract often returns wrapped desktop icon labels as separate line numbers.
    We first combine words from the same OCR line, then merge neighboring lines
    with similar horizontal centers to keep labels like "Visual Studio Code"
    together.
    """

    line_groups: dict[tuple[int, int, int], list[OcrWord]] = defaultdict(list)
    for word in words:
        line_groups[(word.block_num, word.par_num, word.line_num)].append(word)

    lines = [
        _combine_words(cluster)
        for group in line_groups.values()
        for cluster in _split_words_by_horizontal_gap(
            group,
            max_horizontal_gap=max_horizontal_gap,
        )
    ]
    lines.sort(key=lambda line: (line.box.y1, line.box.x1))

    merged: list[OcrLine] = []
    for line in lines:
        if not merged:
            merged.append(line)
            continue

        previous = merged[-1]
        vertical_gap = line.box.y1 - previous.box.y2
        center_delta = abs(_center_x(line.box) - _center_x(previous.box))
        if 0 <= vertical_gap <= max_vertical_gap and center_delta <= max_center_delta:
            merged[-1] = _merge_lines(previous, line)
        else:
            merged.append(line)

    return merged


def _split_words_by_horizontal_gap(
    words: list[OcrWord],
    *,
    max_horizontal_gap: int,
) -> list[list[OcrWord]]:
    ordered = sorted(words, key=lambda word: (word.box.y1, word.box.x1))
    clusters: list[list[OcrWord]] = []
    for word in ordered:
        if not clusters:
            clusters.append([word])
            continue

        previous = clusters[-1][-1]
        gap = word.box.x1 - previous.box.x2
        if gap > max_horizontal_gap:
            clusters.append([word])
        else:
            clusters[-1].append(word)

    return clusters


def _combine_words(words: list[OcrWord]) -> OcrLine:
    ordered = sorted(words, key=lambda word: (word.box.y1, word.box.x1))
    text = " ".join(word.text for word in ordered)
    confidence = sum(word.confidence for word in ordered) / len(ordered)
    box = _union_boxes([word.box for word in ordered], label=text)
    return OcrLine(text=text, confidence=confidence, box=box, word_count=len(ordered))


def _merge_lines(first: OcrLine, second: OcrLine) -> OcrLine:
    text = f"{first.text} {second.text}"
    word_count = first.word_count + second.word_count
    confidence = (
        (first.confidence * first.word_count)
        + (second.confidence * second.word_count)
    ) / word_count
    box = _union_boxes([first.box, second.box], label=text)
    return OcrLine(text=text, confidence=confidence, box=box, word_count=word_count)


def _union_boxes(boxes: list[Box], *, label: str) -> Box:
    return Box(
        min(box.x1 for box in boxes),
        min(box.y1 for box in boxes),
        max(box.x2 for box in boxes),
        max(box.y2 for box in boxes),
        label,
    )


def _parse_confidence(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1


def _scale_coordinate(value: Any, coordinate_scale: int) -> int:
    return int(round(int(value) / max(coordinate_scale, 1)))


def _center_x(box: Box) -> float:
    return (box.x1 + box.x2) / 2
