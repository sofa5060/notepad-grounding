import pytest
from PIL import Image

from notepad_grounding_paper_v2 import llm
from notepad_grounding_paper_v2 import locate
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


def test_build_grid_cells_tiles_the_box_exactly():
    cells = locate.build_grid_cells((0, 0, 400, 300), rows=3, cols=4)
    assert len(cells) == 12
    assert cells[0].id == "R1C1" and cells[0].box == (0, 0, 100, 100)
    assert cells[-1].id == "R3C4" and cells[-1].box == (300, 200, 400, 300)
    assert cells[0].center == (50, 50)


def test_cell_by_id_raises_on_unknown_id():
    cells = locate.build_grid_cells((0, 0, 100, 100), rows=2, cols=2)
    assert locate.cell_by_id(cells, "R2C1").row == 2
    with pytest.raises(ValueError, match="Unknown cell"):
        locate.cell_by_id(cells, "R9C9")


def test_expand_box_clamps_to_bounds():
    assert locate.expand_box((10, 10, 50, 50), padding=20, bounds=(0, 0, 60, 60)) == (0, 0, 60, 60)
    assert locate.expand_box((30, 30, 40, 40), padding=5, bounds=(0, 0, 100, 100)) == (25, 25, 45, 45)


def test_crop_around_point_clamps_to_edges():
    image = Image.new("RGB", (100, 100))
    (crop, box) = locate.crop_around_point(image, center=(5, 95), size=(40, 40))
    assert box == (0, 60, 40, 100)
    assert crop.size == (40, 40)


def test_draw_functions_write_artifacts(tmp_path):
    image = Image.new("RGB", (100, 100))
    cells = locate.build_grid_cells((0, 0, 100, 100), rows=2, cols=2)
    grid = locate.draw_grid(image, cells, output_path=tmp_path / "grid.png", selected_cell_id="R1C1")
    click = locate.draw_click_grid(image, cells, output_path=tmp_path / "click.png", selected_cell_id="R2C2")
    marker = locate.draw_click_marker(image, point=(50, 50), output_path=tmp_path / "marker.png")
    assert grid.size == (100, 100)
    assert click.size == (128, 128)  # gutter=28 added on both axes
    assert marker.size == (100, 100)
    assert (tmp_path / "grid.png").exists() and (tmp_path / "click.png").exists() and (tmp_path / "marker.png").exists()
