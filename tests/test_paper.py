import pytest
from PIL import Image

from notepad_grounding_paper import llm
from notepad_grounding_paper import locate
from notepad_grounding_paper.models import CellChoice
from notepad_grounding_paper.models import TargetReview


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


def fake_first_valid_choice(*, query, image, cell_ids, rejected=(), reviewer_rationale="", style="zoom"):
    cell_id = next(c for c in cell_ids if c not in rejected)
    return CellChoice(cell_id=cell_id, confidence=0.9, rationale="pick first non-rejected")


def test_choose_reviewed_cell_retries_with_rejections(monkeypatch, tmp_path):
    cells = locate.build_grid_cells((0, 0, 100, 100), rows=2, cols=2)
    rejected_seen = []

    def tracking_choose(*, query, image, cell_ids, rejected=(), reviewer_rationale="", style="zoom"):
        rejected_seen.append(list(rejected))
        return fake_first_valid_choice(
            query=query, image=image, cell_ids=cell_ids, rejected=rejected, reviewer_rationale=reviewer_rationale
        )

    reviews = [
        TargetReview(contains_target=False, confidence=0.8, rationale="empty cell"),
        TargetReview(contains_target=True, confidence=0.9, rationale="icon visible"),
    ]
    monkeypatch.setattr(llm, "choose_cell", tracking_choose)
    monkeypatch.setattr(llm, "review_target", lambda **kwargs: reviews.pop(0))

    cell = locate._choose_reviewed_cell(
        crop=Image.new("RGB", (100, 100)), cells=cells, query="Notepad", style="zoom", output_dir=tmp_path, tag="01"
    )
    assert cell.id == "R1C2"
    assert rejected_seen == [[], ["R1C1"]]


def test_choose_reviewed_cell_raises_when_reviewer_never_approves(monkeypatch, tmp_path):
    cells = locate.build_grid_cells((0, 0, 100, 100), rows=2, cols=2)
    monkeypatch.setattr(llm, "choose_cell", fake_first_valid_choice)
    monkeypatch.setattr(
        llm, "review_target", lambda **kwargs: TargetReview(contains_target=False, confidence=0.9, rationale="nope")
    )
    with pytest.raises(ValueError, match="rejected all attempts"):
        locate._choose_reviewed_cell(
            crop=Image.new("RGB", (100, 100)), cells=cells, query="Notepad", style="zoom", output_dir=tmp_path, tag="01"
        )


def test_run_locate_returns_screen_point_and_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "choose_cell", fake_first_valid_choice)
    monkeypatch.setattr(
        llm, "review_target", lambda **kwargs: TargetReview(contains_target=True, confidence=0.9, rationale="ok")
    )
    image = Image.new("RGB", (1200, 800))
    (x, y) = locate.run_locate(image, query="Notepad", output_root=tmp_path, timestamp="test-run")
    assert 0 <= x < 1200 and 0 <= y < 800
    run_dir = tmp_path / "test-run"
    assert (run_dir / "00-source.png").exists()
    assert (run_dir / "01-grid.png").exists() and (run_dir / "01-selected.png").exists()
    assert (run_dir / "final-crop.png").exists()
    assert (run_dir / "click-01-grid.png").exists() and (run_dir / "click-02-grid.png").exists()
    assert (run_dir / "click-point-full.png").exists()
