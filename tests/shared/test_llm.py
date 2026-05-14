import pytest
from PIL import Image

from notepad_grounding.models import CellChoice
from notepad_grounding.models import ClickGridChoice
from notepad_grounding.models import IconDetection
from notepad_grounding.prompts import build_bbox_initial_prompt
from notepad_grounding.prompts import build_bbox_validation_prompt
from notepad_grounding.reviewers import OpenAIBboxReviewer
from notepad_grounding.vision import parse_click_grid_choice
from notepad_grounding.vision import parse_cell_choice
from notepad_grounding.vision import parse_icon_detection
from notepad_grounding.vision import resolve_openai_model


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


def test_parse_click_grid_choice_accepts_valid_json_and_clamps_confidence():
    choice = parse_click_grid_choice(
        '{"cell_id": "R4C4", "confidence": 1.5, "rationale": "center cell"}',
        valid_cell_ids=["R1C1", "R4C4"],
    )

    assert choice == ClickGridChoice(cell_id="R4C4", confidence=1.0, rationale="center cell")


def test_parse_click_grid_choice_rejects_invalid_cell_id():
    with pytest.raises(ValueError):
        parse_click_grid_choice(
            '{"cell_id": "R9C9", "confidence": 0.8, "rationale": "bad"}',
            valid_cell_ids=["R1C1", "R4C4"],
        )


def test_bbox_prompts_explain_box_center_is_click_target():
    initial = build_bbox_initial_prompt(query="Notepad")
    validation = build_bbox_validation_prompt()

    assert "click target" in initial
    assert "center" in initial
    assert "click target" in validation
    assert "center point" in validation


class FakeOpenAIResponse:
    def __init__(self, *, response_id: str, output_text: str):
        self.id = response_id
        self.output_text = output_text


class FakeResponses:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeOpenAIClient:
    def __init__(self, responses):
        self.responses = FakeResponses(responses)


def test_bbox_reviewer_saves_debug_outputs(tmp_path):
    fake_client = FakeOpenAIClient(
        [
            FakeOpenAIResponse(
                response_id="initial",
                output_text='{"target_visible": true, "icon_bbox": [0, 0, 10, 10], "confidence": 0.6, "rationale": "initial"}',
            ),
            FakeOpenAIResponse(
                response_id="review-1",
                output_text='{"confirmed": false, "corrected_icon_bbox": [5, 6, 25, 26], "confidence": 0.8, "rationale": "shifted"}',
            ),
            FakeOpenAIResponse(
                response_id="review-2",
                output_text='{"confirmed": true, "corrected_icon_bbox": [5, 6, 25, 26], "confidence": 0.9, "rationale": "aligned"}',
            ),
        ]
    )
    reviewer = OpenAIBboxReviewer.__new__(OpenAIBboxReviewer)
    reviewer._client = fake_client
    reviewer._model = "test-model"

    detection = reviewer.review_bbox(
        query="Notepad",
        image=Image.new("RGB", (100, 100), "white"),
        max_iterations=2,
        debug_dir=tmp_path,
    )

    assert detection.icon_bbox == (5, 6, 25, 26)
    assert (tmp_path / "bbox-initial-result.json").exists()
    assert (tmp_path / "bbox-review-01.png").exists()
    assert (tmp_path / "bbox-review-01-result.json").exists()
    assert (tmp_path / "bbox-review-02.png").exists()
    assert (tmp_path / "bbox-review-02-result.json").exists()
    assert (tmp_path / "bbox-final-result.json").exists()
    assert '"response_id": "initial"' in (tmp_path / "bbox-initial-result.json").read_text()
    assert '"raw_output"' in (tmp_path / "bbox-review-01-result.json").read_text()
    assert '"final_detection"' in (tmp_path / "bbox-final-result.json").read_text()
