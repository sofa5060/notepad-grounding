from grid_ocr.grounding import find_candidates
from grid_ocr.grounding import locate_from_lines
from grid_ocr.ocr import OcrLine


def test_find_candidates_places_icon_above_label():
    line = OcrLine(text="Notepad", confidence=100, box=(700, 460, 760, 478))

    candidates = find_candidates([line], screen_size=(1920, 1080), query="Notepad")

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.label_text == "Notepad"
    assert candidate.icon_box[3] <= line.box[1]
    assert candidate.score > 0.8


def test_locate_from_lines_selects_best_fuzzy_query_match():
    lines = [
        OcrLine(text="Recycle Bin", confidence=100, box=(20, 80, 90, 100)),
        OcrLine(text="Note pad", confidence=100, box=(700, 460, 760, 478)),
    ]

    result = locate_from_lines(lines, screen_size=(1920, 1080), query="Notepad")

    assert result is not None
    assert result.label_text == "Note pad"
    assert result.icon_center[1] < result.label_box[1]


def test_low_score_candidates_do_not_locate():
    lines = [
        OcrLine(text="Recycle Bin", confidence=100, box=(20, 80, 90, 100)),
        OcrLine(text="Steam", confidence=100, box=(20, 180, 80, 200)),
    ]

    result = locate_from_lines(lines, screen_size=(1920, 1080), query="Notepad")

    assert result is None
