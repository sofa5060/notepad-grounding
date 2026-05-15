from __future__ import annotations

from PIL import Image

from direct_llm_grounding.notepad_direct import CoordinateGuess
from direct_llm_grounding.notepad_direct import draw_debug_box
from direct_llm_grounding.notepad_direct import parse_coordinate_guess


def test_parse_coordinate_guess_from_json_fence() -> None:
    guess = parse_coordinate_guess(
        """
        ```json
        {
          "x": 123.7,
          "y": 45.2,
          "bbox": {"left": 100, "top": 20, "right": 160, "bottom": 80},
          "confidence": 0.82,
          "rationale": "The Notepad shortcut is in the top-left group."
        }
        ```
        """,
        image_size=(1920, 1080),
    )

    assert guess == CoordinateGuess(
        x=124,
        y=45,
        bbox=(100, 20, 160, 80),
        confidence=0.82,
        rationale="The Notepad shortcut is in the top-left group.",
    )


def test_parse_coordinate_guess_clamps_point_and_bbox() -> None:
    guess = parse_coordinate_guess(
        '{"x": -10, "y": 1200, "bbox": {"left": -20, "top": 900, "right": 3000, "bottom": 1200}}',
        image_size=(1920, 1080),
    )

    assert guess.x == 0
    assert guess.y == 1079
    assert guess.bbox == (0, 900, 1919, 1079)


def test_draw_debug_box_marks_reported_location() -> None:
    image = Image.new("RGB", (200, 120), "white")
    guess = CoordinateGuess(x=50, y=40, bbox=(30, 20, 70, 60), confidence=0.5, rationale="")

    annotated = draw_debug_box(image, guess)

    assert annotated.getpixel((30, 20)) == (255, 0, 0)
    assert annotated.getpixel((50, 40)) == (255, 0, 0)
