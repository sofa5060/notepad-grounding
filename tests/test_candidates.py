from notepad_grounding.grounding.annotations import Box
from notepad_grounding.grounding.candidates import infer_icon_candidates
from notepad_grounding.grounding.ocr import OcrLine


def test_infer_icon_candidate_places_icon_above_and_centered_on_label():
    label = OcrLine("Notepad", 92, Box(699, 457, 746, 475, "Notepad"))

    candidates = infer_icon_candidates([label], screen_size=(1920, 1080))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.candidate_id == 1
    assert candidate.label_text == "Notepad"
    assert candidate.icon_box.y2 <= candidate.label_box.y1
    assert abs(candidate.icon_center[0] - candidate.label_center[0]) <= 1
    assert candidate.combined_box.x1 <= candidate.icon_box.x1
    assert candidate.combined_box.y1 <= candidate.icon_box.y1
    assert candidate.combined_box.x2 >= candidate.label_box.x2
    assert candidate.combined_box.y2 >= candidate.label_box.y2


def test_short_labels_still_get_minimum_useful_icon_box():
    label = OcrLine("kmt", 88, Box(250, 240, 272, 254, "kmt"))

    candidate = infer_icon_candidates([label], screen_size=(1920, 1080))[0]

    assert candidate.icon_box.x2 - candidate.icon_box.x1 >= 32
    assert candidate.icon_box.y2 - candidate.icon_box.y1 >= 32
    assert "min_icon_size=32" in candidate.confidence_notes


def test_long_labels_do_not_create_absurdly_wide_icon_boxes():
    label = OcrLine(
        "Very Long Desktop Shortcut Label",
        88,
        Box(500, 400, 820, 418, "Very Long Desktop Shortcut Label"),
    )

    candidate = infer_icon_candidates([label], screen_size=(1920, 1080))[0]

    assert candidate.icon_box.x2 - candidate.icon_box.x1 <= 160
    assert "max_icon_size=144" in candidate.confidence_notes


def test_taskbar_area_labels_are_ignored():
    label = OcrLine("Search", 95, Box(730, 1048, 780, 1065, "Search"))

    candidates = infer_icon_candidates([label], screen_size=(1920, 1080))

    assert candidates == []


def test_non_alphanumeric_ocr_fragments_are_ignored():
    label = OcrLine("=", 95, Box(200, 200, 220, 214, "="))

    candidates = infer_icon_candidates([label], screen_size=(1920, 1080))

    assert candidates == []


def test_icon_boxes_clamp_to_screen_edges():
    label = OcrLine("Edge", 95, Box(2, 120, 38, 136, "Edge"))

    candidate = infer_icon_candidates([label], screen_size=(1920, 1080))[0]

    assert candidate.icon_box.x1 == 0
    assert candidate.combined_box.x1 == 0
