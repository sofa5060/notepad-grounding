from PIL import Image

from grid_ocr.ocr import OcrLine
from grid_ocr.ocr import OcrWord
from grid_ocr.ocr import dedupe_ocr_words
from grid_ocr.ocr import extract_ocr_lines_from_grid
from grid_ocr.ocr import group_words_by_line
from grid_ocr.ocr import iter_ocr_tiles
from grid_ocr.ocr import offset_and_scale_words
from grid_ocr.ocr import prepare_ocr_image
from grid_ocr.ocr import scale_ocr_lines


def test_group_words_by_line_merges_wrapped_vertical_icon_label():
    words = [
        OcrWord(text="Visual", confidence=100, box=(20, 900, 58, 914), line_id=1),
        OcrWord(text="Studio", confidence=100, box=(18, 918, 58, 932), line_id=2),
        OcrWord(text="Code", confidence=100, box=(24, 936, 54, 950), line_id=3),
    ]

    lines = group_words_by_line(words)

    assert lines == [
        OcrLine(text="Visual Studio Code", confidence=100, box=(18, 900, 58, 950))
    ]


def test_group_words_by_line_merges_wrapped_label_when_centers_are_shifted():
    words = [
        OcrWord(text="Screen", confidence=100, box=(100, 500, 145, 516), line_id=1),
        OcrWord(text="Recorder", confidence=100, box=(150, 520, 235, 536), line_id=2),
    ]

    lines = group_words_by_line(words)

    assert lines == [
        OcrLine(text="Screen Recorder", confidence=100, box=(100, 500, 235, 536))
    ]


def test_group_words_by_line_does_not_merge_vertically_close_distant_labels():
    words = [
        OcrWord(text="Notepad", confidence=100, box=(100, 500, 160, 516), line_id=1),
        OcrWord(text="Terminal", confidence=100, box=(240, 520, 310, 536), line_id=2),
    ]

    lines = group_words_by_line(words)

    assert [line.text for line in lines] == ["Notepad", "Terminal"]


def test_group_words_by_line_keeps_nearby_horizontal_icon_labels_separate():
    words = [
        OcrWord(text="Notepad", confidence=100, box=(700, 460, 760, 478), line_id=1),
        OcrWord(text="Terminal", confidence=100, box=(790, 461, 850, 479), line_id=1),
    ]

    lines = group_words_by_line(words, max_horizontal_gap=16)

    assert [line.text for line in lines] == ["Notepad", "Terminal"]


def test_prepare_ocr_image_upscales_without_changing_original():
    image = Image.new("RGB", (100, 50), "white")

    scaled = prepare_ocr_image(image, scale=2)

    assert image.size == (100, 50)
    assert scaled.size == (200, 100)


def test_scale_ocr_lines_maps_scaled_boxes_back_to_screen_coordinates():
    lines = [
        OcrLine(text="Notepad", confidence=100, box=(200, 100, 300, 140)),
    ]

    scaled = scale_ocr_lines(lines, scale=2)

    assert scaled == [
        OcrLine(text="Notepad", confidence=100, box=(100, 50, 150, 70)),
    ]


def test_iter_ocr_tiles_uses_overlap_and_covers_screen():
    tiles = iter_ocr_tiles(width=220, height=120, tile_width=100, tile_height=80, overlap=20)

    assert tiles[0].box == (0, 0, 100, 80)
    assert tiles[1].box == (80, 0, 180, 80)
    assert tiles[-1].box == (120, 40, 220, 120)


def test_offset_and_scale_words_maps_tile_ocr_to_screen_coordinates():
    words = [
        OcrWord(text="Notepad", confidence=100, box=(20, 10, 120, 40), line_id=1),
    ]

    mapped = offset_and_scale_words(words, offset_x=400, offset_y=300, scale=2, line_id_offset=100)

    assert mapped == [
        OcrWord(text="Notepad", confidence=100, box=(410, 305, 460, 320), line_id=101),
    ]


def test_dedupe_ocr_words_removes_duplicate_words_from_overlapping_tiles():
    words = [
        OcrWord(text="Notepad", confidence=90, box=(100, 100, 160, 118), line_id=1),
        OcrWord(text="Notepad", confidence=100, box=(102, 101, 162, 119), line_id=2),
        OcrWord(text="Terminal", confidence=100, box=(220, 100, 290, 118), line_id=3),
    ]

    deduped = dedupe_ocr_words(words)

    assert deduped == [
        OcrWord(text="Notepad", confidence=100, box=(102, 101, 162, 119), line_id=2),
        OcrWord(text="Terminal", confidence=100, box=(220, 100, 290, 118), line_id=3),
    ]


def test_extract_ocr_lines_from_grid_merges_words_across_neighboring_tiles():
    image = Image.new("RGB", (220, 120), "white")

    def fake_ocr(tile_image):
        if tile_image.size != (200, 160):
            return []
        if len(calls) == 0:
            calls.append(tile_image.size)
            return [OcrWord(text="Note", confidence=100, box=(160, 60, 196, 84), line_id=1)]
        if len(calls) == 1:
            calls.append(tile_image.size)
            return [OcrWord(text="pad", confidence=100, box=(4, 60, 46, 84), line_id=1)]
        calls.append(tile_image.size)
        return []

    calls = []

    lines = extract_ocr_lines_from_grid(
        image,
        ocr_words=fake_ocr,
        tile_width=100,
        tile_height=80,
        overlap=20,
        scale=2,
    )

    assert lines[0].text == "Note pad"
    assert lines[0].box == (80, 30, 103, 42)
