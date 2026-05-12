from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
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
) -> list[OcrLine]:
    """Run Tesseract OCR and return grouped desktop-label-like text lines."""

    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as exc:
        raise OcrError("Missing OCR dependency 'pytesseract'. Run 'uv sync'.") from exc

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

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
            "--tesseract-cmd with the executable path."
        ) from exc

    words = extract_words_from_tesseract_data(
        data,
        min_confidence=min_confidence,
        coordinate_scale=upscale_factor,
    )
    return group_words_by_line(words)


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
