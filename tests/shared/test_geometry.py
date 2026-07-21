from notepad_grounding_paper.images import build_grid_cells
from notepad_grounding_paper.images import expand_box


def test_build_grid_cells_returns_labeled_cells_covering_box():
    cells = build_grid_cells((10, 20, 210, 120), rows=2, cols=4, prefix="R1-")

    assert cells[0].id == "R1-1-1"
    assert cells[0].box == (10, 20, 60, 70)
    assert cells[-1].id == "R1-2-4"
    assert cells[-1].box == (160, 70, 210, 120)


def test_expand_box_clamps_to_bounds():
    assert expand_box((10, 10, 30, 30), padding=20, bounds=(0, 0, 100, 100)) == (0, 0, 50, 50)
