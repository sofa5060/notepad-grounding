import pytest

from notepad_grounding.shared.llm import CellChoice
from notepad_grounding.shared.llm import IconDetection
from notepad_grounding.shared.llm import parse_cell_choice
from notepad_grounding.shared.llm import parse_icon_detection
from notepad_grounding.shared.llm import resolve_openai_model


def test_parse_cell_choice_accepts_valid_json():
    choice = parse_cell_choice(
        '{"cell_id": "R1-1-2", "confidence": 0.8, "rationale": "icon is visible"}',
        valid_cell_ids=["R1-1-1", "R1-1-2"],
    )

    assert choice == CellChoice(cell_id="R1-1-2", confidence=0.8, rationale="icon is visible")


def test_parse_cell_choice_rejects_invalid_cell_id():
    with pytest.raises(ValueError):
        parse_cell_choice(
            '{"cell_id": "bad", "confidence": 0.8, "rationale": "nope"}',
            valid_cell_ids=["R1-1-1"],
        )


def test_resolve_openai_model_defaults_to_gpt_5_4(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert resolve_openai_model() == "gpt-5.4"


def test_resolve_openai_model_prefers_explicit_argument(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "from-env")

    assert resolve_openai_model("explicit-model") == "explicit-model"


def test_parse_icon_detection_clamps_crop_local_bbox():
    detection = parse_icon_detection(
        '{"target_visible": true, "icon_bbox": [-5, 10, 55, 70], "confidence": 0.9, "rationale": "icon"}',
        image_size=(50, 60),
    )

    assert detection == IconDetection(
        target_visible=True,
        icon_bbox=(0, 10, 50, 60),
        confidence=0.9,
        rationale="icon",
    )
