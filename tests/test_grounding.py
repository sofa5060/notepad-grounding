from notepad_grounding.grounding import build_grid
from notepad_grounding.grounding import infer_candidates
from notepad_grounding.grounding import locate_from_lines
from notepad_grounding.ocr import OcrLine


def test_build_grid_covers_screen_above_taskbar():
    cells = build_grid((1920, 1080), cell_width=96, cell_height=96, taskbar_height=48)

    assert cells[0].box == (0, 0, 96, 96)
    assert cells[-1].box[3] <= 1032
    assert len(cells) > 100


def test_infer_candidates_places_icon_above_label():
    line = OcrLine(text="Notepad", confidence=100, box=(700, 460, 760, 478))

    candidates = infer_candidates([line], screen_size=(1920, 1080), query="Notepad")

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
    assert result.candidate.label_text == "Note pad"
    assert result.center == result.candidate.icon_center


def test_low_score_candidates_do_not_locate():
    lines = [
        OcrLine(text="Recycle Bin", confidence=100, box=(20, 80, 90, 100)),
        OcrLine(text="Steam", confidence=100, box=(20, 180, 80, 200)),
    ]

    result = locate_from_lines(lines, screen_size=(1920, 1080), query="Notepad")

    assert result is None
