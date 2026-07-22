import pytest
from PIL import Image

import notepad_grounding_paper.models as models
from notepad_grounding_paper.models import CellChoice
from notepad_grounding_paper.prompts import build_cell_choice_prompt
from notepad_grounding_paper.prompts import build_choice_correction_prompt
from notepad_grounding_paper.prompts import build_revise_cell_choice_prompt
from notepad_grounding_paper.llm import OpenAIVisionClient
from notepad_grounding_paper.llm import parse_cell_choice


def test_parse_cell_choice_accepts_valid_json():
    choice = parse_cell_choice(
        '{"cell_id": "R1-1-2", "confidence": 0.8, "rationale": "icon is visible"}', valid_cell_ids=["R1-1-1", "R1-1-2"]
    )

    assert choice == CellChoice(cell_id="R1-1-2", confidence=0.8, rationale="icon is visible")


def test_parse_cell_choice_rejects_invalid_cell_id():
    with pytest.raises(ValueError):
        parse_cell_choice('{"cell_id": "bad", "confidence": 0.8, "rationale": "nope"}', valid_cell_ids=["R1-1-1"])


def test_parse_cell_choice_accepts_click_grid_ids_and_clamps_confidence():
    choice = parse_cell_choice(
        '{"cell_id": "R4C4", "confidence": 1.5, "rationale": "center cell"}', valid_cell_ids=["R1C1", "R4C4"]
    )

    assert choice == CellChoice(cell_id="R4C4", confidence=1.0, rationale="center cell")


def test_parse_cell_choice_rejects_invalid_click_grid_id():
    with pytest.raises(ValueError):
        parse_cell_choice('{"cell_id": "R9C9", "confidence": 0.8, "rationale": "bad"}', valid_cell_ids=["R1C1", "R4C4"])


def test_click_grid_choice_uses_normal_cell_choice_model():
    assert not hasattr(models, "ClickGridChoice")


def test_cell_choice_prompts_are_built_outside_vision_client():
    initial = build_cell_choice_prompt(query="Notepad", cell_ids=["R1-1-1", "R1-1-2"])
    revision = build_revise_cell_choice_prompt(
        query="Notepad", rejected_cell_ids=["R1-1-1"], reviewer_rationale="wrong crop", valid_cell_ids=["R1-1-2"]
    )
    correction = build_choice_correction_prompt(error="invalid cell", valid_cell_ids=["R1-1-2"])

    assert "Notepad" in initial
    assert "R1-1-1, R1-1-2" in initial
    assert "Do NOT choose any rejected cell: R1-1-1" in revision
    assert "wrong crop" in revision
    assert "Problem: invalid cell" in correction
    assert "R1-1-2" in correction


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


def test_revise_cell_choice_retries_invalid_cell_id():
    fake_client = FakeOpenAIClient(
        [
            FakeOpenAIResponse(
                response_id="bad-choice",
                output_text='{"cell_id": "R1-1-1", "confidence": 0.8, "rationale": "old cell"}',
            ),
            FakeOpenAIResponse(
                response_id="good-choice",
                output_text='{"cell_id": "R1-1-2", "confidence": 0.7, "rationale": "corrected"}',
            ),
        ]
    )
    client = OpenAIVisionClient.__new__(OpenAIVisionClient)
    client._client = fake_client
    client._model = "test-model"

    choice = client.revise_cell_choice(
        query="Notepad",
        image=Image.new("RGB", (20, 20), "white"),
        cell_ids=["R1-1-1", "R1-1-2"],
        rejected_cell_ids=["R1-1-1"],
        reviewer_rationale="wrong crop",
        previous_response_id="previous",
    )

    assert choice == CellChoice(cell_id="R1-1-2", confidence=0.7, rationale="corrected", response_id="good-choice")
    assert fake_client.responses.calls[0]["previous_response_id"] == "previous"
    assert fake_client.responses.calls[1]["previous_response_id"] == "bad-choice"


def test_choose_click_grid_cell_retries_invalid_cell_id():
    fake_client = FakeOpenAIClient(
        [
            FakeOpenAIResponse(
                response_id="bad-choice", output_text='{"cell_id": "R1C1", "confidence": 0.9, "rationale": "rejected"}'
            ),
            FakeOpenAIResponse(
                response_id="good-choice", output_text='{"cell_id": "R2C2", "confidence": 0.8, "rationale": "center"}'
            ),
        ]
    )
    client = OpenAIVisionClient.__new__(OpenAIVisionClient)
    client._client = fake_client
    client._model = "test-model"

    choice = client.choose_click_grid_cell(
        query="Notepad",
        image=Image.new("RGB", (20, 20), "white"),
        cell_ids=["R1C1", "R2C2"],
        rejected_cell_ids=["R1C1"],
        previous_response_id="previous",
    )

    assert choice == CellChoice(cell_id="R2C2", confidence=0.8, rationale="center", response_id="good-choice")
    assert fake_client.responses.calls[0]["previous_response_id"] == "previous"
    assert fake_client.responses.calls[1]["previous_response_id"] == "bad-choice"


def test_choice_retry_loop_stops_after_two_retries():
    fake_client = FakeOpenAIClient(
        [
            FakeOpenAIResponse(
                response_id="bad-choice-1", output_text='{"cell_id": "bad", "confidence": 0.8, "rationale": "wrong"}'
            ),
            FakeOpenAIResponse(
                response_id="bad-choice-2",
                output_text='{"cell_id": "bad", "confidence": 0.8, "rationale": "wrong again"}',
            ),
            FakeOpenAIResponse(
                response_id="bad-choice-3",
                output_text='{"cell_id": "bad", "confidence": 0.8, "rationale": "still wrong"}',
            ),
        ]
    )
    client = OpenAIVisionClient.__new__(OpenAIVisionClient)
    client._client = fake_client
    client._model = "test-model"

    with pytest.raises(ValueError, match="failed to return a valid cell_id after 2 retries"):
        client.choose_click_grid_cell(query="Notepad", image=Image.new("RGB", (20, 20), "white"), cell_ids=["R1C1"])

    assert len(fake_client.responses.calls) == 3
