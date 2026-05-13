from PIL import Image

from notepad_grounding.flows.llm_visual_search.flow import run_llm_visual_search
from notepad_grounding.shared.llm import CellChoice
from notepad_grounding.shared.llm import CellsChoice


class FakeVisionClient:
    def __init__(self, choices, final_cells):
        self.choices = list(choices)
        self.final_cells = final_cells
        self.calls = []

    def choose_cell(self, *, query, image, cell_ids):
        self.calls.append(("choose_cell", query, image.size, cell_ids))
        return CellChoice(cell_id=self.choices.pop(0), confidence=0.9, rationale="test choice")

    def choose_cells(self, *, query, image, cell_ids):
        self.calls.append(("choose_cells", query, image.size, cell_ids))
        return CellsChoice(cell_ids=self.final_cells, confidence=0.95, rationale="final cells")

    def locate_icon(self, *, query, image):
        self.calls.append(("locate_icon", query, image.size, ["final"]))
        raise NotImplementedError("locate_icon should not be called in this flow")


def test_run_llm_visual_search_maps_selected_cells_to_screen_coordinates(tmp_path):
    image = Image.new("RGB", (400, 300), "white")
    client = FakeVisionClient(
        ["R1-2-2", "R2-1-1"],
        ["F-2-2", "F-2-3", "F-3-2", "F-3-3"],
    )

    result = run_llm_visual_search(
        image,
        query="Notepad",
        client=client,
        output_root=tmp_path,
        timestamp="run",
        rounds=2,
        first_grid=(2, 2),
        later_grid=(2, 2),
        crop_padding=0,
        final_crop_max_size=(50, 50),
    )

    # Final crop after 2 rounds is (200,150,300,225) -> 100x75.
    # Fine grid 5x5 gives cells of 20x15.  The four centre cells F-2-2, F-2-3,
    # F-3-2, F-3-3 union to local box (20,15,60,45) -> screen (220,165,260,195).
    # Centre of that box is (240,180).
    assert result.center == (240, 180)
    assert result.final_box == (220, 165, 260, 195)
    assert result.final_detection.icon_bbox_local == (20, 15, 60, 45)
    assert result.final_detection.icon_bbox_screen == (220, 165, 260, 195)
    assert len(result.steps) == 2
    assert (tmp_path / "run" / "01-selected.png").exists()
    assert (tmp_path / "run" / "02-selected.png").exists()
    assert (tmp_path / "run" / "final-crop.png").exists()
    assert (tmp_path / "run" / "final-grid.png").exists()
    assert (tmp_path / "run" / "final-selected.png").exists()
    assert (tmp_path / "run" / "result.json").exists()
    assert client.calls[0][1] == "Notepad"
    assert client.calls[0][3] == ["R1-1-1", "R1-1-2", "R1-2-1", "R1-2-2"]
