import pytest
from PIL import Image

from notepad_grounding_paper_v2 import llm
from notepad_grounding_paper_v2.models import CellChoice


def test_choose_cell_retries_on_invalid_cell_id(monkeypatch):
    answers = [
        CellChoice(cell_id="bogus", confidence=0.9, rationale="oops"),
        CellChoice(cell_id="R1C2", confidence=0.9, rationale="better"),
    ]
    prompts = []

    def fake_ask(prompt, image, response_model):
        prompts.append(prompt)
        return answers.pop(0)

    monkeypatch.setattr(llm, "_ask", fake_ask)
    choice = llm.choose_cell(query="Notepad", image=Image.new("RGB", (10, 10)), cell_ids=["R1C1", "R1C2"])
    assert choice.cell_id == "R1C2"
    assert len(prompts) == 2
    assert "'bogus'" in prompts[1]


def test_choose_cell_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(llm, "_ask", lambda *a, **k: CellChoice(cell_id="bogus", confidence=0.5, rationale="no"))
    with pytest.raises(ValueError, match="valid cell_id"):
        llm.choose_cell(query="Notepad", image=Image.new("RGB", (10, 10)), cell_ids=["R1C1"])


def test_choose_cell_puts_rejections_in_the_prompt(monkeypatch):
    prompts = []

    def fake_ask(prompt, image, response_model):
        prompts.append(prompt)
        return CellChoice(cell_id="R1C2", confidence=0.9, rationale="ok")

    monkeypatch.setattr(llm, "_ask", fake_ask)
    choice = llm.choose_cell(
        query="Notepad",
        image=Image.new("RGB", (10, 10)),
        cell_ids=["R1C1", "R1C2"],
        rejected=["R1C1"],
        reviewer_rationale="wrong icon",
    )
    assert choice.cell_id == "R1C2"
    assert "Do NOT choose these rejected cells: R1C1" in prompts[0]
    assert "wrong icon" in prompts[0]
    assert "Valid cell_id values are: R1C2" in prompts[0]


def test_choose_cell_rejects_answers_from_the_rejected_list(monkeypatch):
    answers = [
        CellChoice(cell_id="R1C1", confidence=0.9, rationale="insists on rejected cell"),
        CellChoice(cell_id="R1C2", confidence=0.9, rationale="fine"),
    ]
    monkeypatch.setattr(llm, "_ask", lambda *a, **k: answers.pop(0))
    choice = llm.choose_cell(
        query="Notepad", image=Image.new("RGB", (10, 10)), cell_ids=["R1C1", "R1C2"], rejected=["R1C1"]
    )
    assert choice.cell_id == "R1C2"
