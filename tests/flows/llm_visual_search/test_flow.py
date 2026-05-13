from PIL import Image

from notepad_grounding.flows.llm_visual_search.flow import run_llm_visual_search
from notepad_grounding.shared.llm import CellChoice
from notepad_grounding.shared.llm import IconDetection


class FakeVisionClient:
    def __init__(self, choices, detection):
        self.choices = list(choices)
        self.detection = detection
        self.calls = []

    def choose_cell(self, *, query, image, cell_ids):
        self.calls.append((query, image.size, cell_ids))
        return CellChoice(cell_id=self.choices.pop(0), confidence=0.9, rationale="test choice")

    def locate_icon(self, *, query, image):
        self.calls.append((query, image.size, ["final"]))
        return self.detection


def test_run_llm_visual_search_maps_selected_cells_to_screen_coordinates(tmp_path):
    image = Image.new("RGB", (400, 300), "white")
    client = FakeVisionClient(
        ["R1-2-2", "R2-1-1"],
        IconDetection(
            target_visible=True,
            icon_bbox=(20, 10, 60, 50),
            confidence=0.92,
            rationale="icon is visible",
        ),
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

    assert result.center == (240, 180)
    assert result.final_box == (220, 160, 260, 200)
    assert result.final_detection.icon_bbox_local == (20, 10, 60, 50)
    assert result.final_detection.icon_bbox_screen == (220, 160, 260, 200)
    assert len(result.steps) == 2
    assert (tmp_path / "run" / "01-selected.png").exists()
    assert (tmp_path / "run" / "02-selected.png").exists()
    assert (tmp_path / "run" / "final-crop.png").exists()
    assert (tmp_path / "run" / "final-detection.png").exists()
    assert (tmp_path / "run" / "result.json").exists()
    assert client.calls[0][0] == "Notepad"
    assert client.calls[0][2] == ["R1-1-1", "R1-1-2", "R1-2-1", "R1-2-2"]
