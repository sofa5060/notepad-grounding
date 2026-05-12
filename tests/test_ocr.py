from PIL import Image

from notepad_grounding.grounding.annotations import Box
from notepad_grounding.grounding.ocr import (
    OcrWord,
    dedupe_ocr_words,
    extract_words_from_tesseract_data,
    group_words_by_line,
    iter_ocr_tiles,
    offset_ocr_words,
    prepare_desktop_label_image,
    windows_ocr_result_to_words,
)


def test_extract_words_from_tesseract_data_ignores_blank_and_low_confidence_rows():
    data = {
        "text": ["", "Notepad", "noise"],
        "conf": ["-1", "92.5", "20"],
        "left": [0, 699, 10],
        "top": [0, 457, 10],
        "width": [0, 47, 20],
        "height": [0, 18, 10],
        "block_num": [0, 1, 1],
        "par_num": [0, 1, 1],
        "line_num": [0, 1, 2],
        "word_num": [0, 1, 1],
    }

    words = extract_words_from_tesseract_data(data, min_confidence=50)

    assert words == [
        OcrWord(
            text="Notepad",
            confidence=92.5,
            box=Box(699, 457, 746, 475, "Notepad"),
            block_num=1,
            par_num=1,
            line_num=1,
            word_num=1,
        )
    ]


def test_extract_words_from_tesseract_data_rescales_coordinates():
    data = {
        "text": ["otepad"],
        "conf": ["90"],
        "left": [1420],
        "top": [922],
        "width": [69],
        "height": [24],
        "block_num": [1],
        "par_num": [1],
        "line_num": [1],
        "word_num": [1],
    }

    words = extract_words_from_tesseract_data(
        data,
        min_confidence=50,
        coordinate_scale=2,
    )

    assert words[0].box == Box(710, 461, 744, 473, "otepad")


def test_group_words_by_line_combines_multi_word_labels():
    words = [
        OcrWord("Visual", 91, Box(4, 958, 45, 972, "Visual"), 1, 1, 1, 1),
        OcrWord("Studio", 90, Box(6, 973, 50, 987, "Studio"), 1, 1, 2, 1),
        OcrWord("Code", 89, Box(12, 988, 44, 1002, "Code"), 1, 1, 3, 1),
    ]

    lines = group_words_by_line(words, max_vertical_gap=4, max_center_delta=32)

    assert len(lines) == 1
    assert lines[0].text == "Visual Studio Code"
    assert lines[0].box == Box(4, 958, 50, 1002, "Visual Studio Code")
    assert lines[0].confidence == 90


def test_group_words_by_line_splits_far_apart_words_on_same_ocr_line():
    words = [
        OcrWord("Notepad", 91, Box(700, 460, 748, 476, "Notepad"), 1, 1, 1, 1),
        OcrWord("Steam", 90, Box(20, 360, 58, 376, "Steam"), 1, 1, 1, 2),
    ]

    lines = group_words_by_line(words, max_horizontal_gap=48)

    assert [line.text for line in lines] == ["Steam", "Notepad"]


def test_group_words_by_line_keeps_close_neighbors_separate_when_gap_is_tight():
    words = [
        OcrWord("CleanShot", 91, Box(90, 260, 148, 274, "CleanShot"), 1, 1, 1, 1),
        OcrWord("2025-08", 90, Box(154, 260, 210, 274, "2025-08"), 1, 1, 1, 2),
    ]

    lines = group_words_by_line(words, max_horizontal_gap=4)

    assert [line.text for line in lines] == ["CleanShot", "2025-08"]


def test_iter_ocr_tiles_covers_image_with_overlap():
    tiles = iter_ocr_tiles(width=700, height=500, tile_size=300, overlap=80)

    assert tiles[0] == Box(0, 0, 300, 300, "tile")
    assert tiles[-1] == Box(400, 200, 700, 500, "tile")
    assert all(tile.x2 - tile.x1 <= 300 for tile in tiles)
    assert all(tile.y2 - tile.y1 <= 300 for tile in tiles)


def test_offset_ocr_words_maps_tile_coordinates_to_screen_coordinates():
    words = [
        OcrWord("Notepad", 91, Box(20, 30, 70, 46, "Notepad"), 1, 1, 1, 1),
    ]

    offset = offset_ocr_words(words, offset_x=400, offset_y=200, group_offset=100)

    assert offset[0].box == Box(420, 230, 470, 246, "Notepad")
    assert offset[0].block_num == 101


def test_dedupe_ocr_words_keeps_highest_confidence_duplicate():
    words = [
        OcrWord("Notepad", 82, Box(420, 230, 470, 246, "Notepad"), 1, 1, 1, 1),
        OcrWord("Notepad", 94, Box(422, 231, 471, 247, "Notepad"), 2, 1, 1, 1),
        OcrWord("Steam", 90, Box(20, 360, 58, 376, "Steam"), 3, 1, 1, 1),
    ]

    deduped = dedupe_ocr_words(words)

    assert [word.text for word in deduped] == ["Notepad", "Steam"]
    assert deduped[0].confidence == 94


def test_windows_ocr_result_to_words_maps_lines_and_bounding_rects():
    class Rect:
        def __init__(self, x, y, width, height):
            self.x = x
            self.y = y
            self.width = width
            self.height = height

    class Word:
        def __init__(self, text, rect):
            self.text = text
            self.bounding_rect = rect

    class Line:
        def __init__(self, words):
            self.words = words

    class Result:
        lines = [
            Line(
                [
                    Word("Note", Rect(700, 450, 30, 14)),
                    Word("pad", Rect(732, 450, 20, 14)),
                ]
            )
        ]

    words = windows_ocr_result_to_words(Result())

    assert words == [
        OcrWord("Note", 100, Box(700, 450, 730, 464, "Note"), 1, 1, 1, 1),
        OcrWord("pad", 100, Box(732, 450, 752, 464, "pad"), 1, 1, 1, 2),
    ]


def test_prepare_desktop_label_image_amplifies_white_text_on_blue_background():
    # Create a synthetic desktop-like image: blue background with a white text region
    image = Image.new("RGB", (100, 60), "#2B7BF6")  # Windows-ish blue wallpaper
    pixels = image.load()
    # Draw a small "white text" block (simulating icon label)
    for x in range(20, 80):
        for y in range(25, 35):
            pixels[x, y] = (240, 240, 240)  # White text
    # Add shadow below
    for x in range(20, 80):
        for y in range(35, 38):
            pixels[x, y] = (30, 30, 30)  # Dark shadow

    preprocessed = prepare_desktop_label_image(image, upscale_factor=1)

    # Preprocessed image should be grayscale
    assert preprocessed.mode == "L"
    # Text region should be bright (amplified)
    text_region = preprocessed.getpixel((50, 30))
    bg_region = preprocessed.getpixel((10, 10))
    shadow_region = preprocessed.getpixel((50, 36))
    # Text should be brighter than background
    assert text_region > bg_region
    # Shadow should be darker than background (or at least different)
    assert shadow_region != bg_region


def test_prepare_desktop_label_image_upscales_output():
    image = Image.new("RGB", (100, 60), "#2B7BF6")
    preprocessed = prepare_desktop_label_image(image, upscale_factor=3)

    assert preprocessed.size == (300, 180)
