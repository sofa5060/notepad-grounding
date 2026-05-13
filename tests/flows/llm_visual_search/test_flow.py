import json

import pytest
from PIL import Image

from notepad_grounding.flows.llm_visual_search.flow import run_llm_visual_search
from notepad_grounding.shared.llm import CellChoice
from notepad_grounding.shared.llm import IconDetection
from notepad_grounding.shared.grid_judge import GridJudgeResult


class FakeVisionClient:
    def __init__(self, choices, detection):
        self.choices = list(choices)
        self.detection = detection
        self.calls = []

    def choose_cell(self, *, query, image, cell_ids):
        self.calls.append(("choose_cell", query, image.size, cell_ids))
        index = len([call for call in self.calls if call[0] in {"choose_cell", "revise_cell_choice"}])
        return CellChoice(
            cell_id=self.choices.pop(0),
            confidence=0.9,
            rationale="test choice",
            response_id=f"response-{index}",
        )

    def revise_cell_choice(
        self,
        *,
        query,
        image,
        cell_ids,
        rejected_cell_ids,
        judge_rationale,
        previous_response_id,
    ):
        self.calls.append(
            (
                "revise_cell_choice",
                query,
                image.size,
                cell_ids,
                list(rejected_cell_ids),
                judge_rationale,
                previous_response_id,
            )
        )
        index = len([call for call in self.calls if call[0] in {"choose_cell", "revise_cell_choice"}])
        return CellChoice(
            cell_id=self.choices.pop(0),
            confidence=0.8,
            rationale="revised choice",
            response_id=f"response-{index}",
        )

    def locate_icon_with_validation(self, *, query, image, max_iterations=3):
        self.calls.append(("locate_icon_with_validation", query, image.size, max_iterations))
        return self.detection


class FakeGridJudge:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = []

    def judge_crop(self, *, query, image):
        self.calls.append((query, image.size))
        contains_target = self.verdicts.pop(0)
        return GridJudgeResult(
            contains_target=contains_target,
            confidence=0.95 if contains_target else 0.85,
            rationale="accepted" if contains_target else "wrong crop",
            visible_evidence="notepad visible" if contains_target else "notepad absent",
        )


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
    assert client.calls[0][1] == "Notepad"
    assert client.calls[0][3] == ["R1-1-1", "R1-1-2", "R1-2-1", "R1-2-2"]


def test_run_llm_visual_search_with_judge_accepts_first_choice_and_saves_artifacts(tmp_path):
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
    judge = FakeGridJudge([True, True])

    result = run_llm_visual_search(
        image,
        query="Notepad",
        client=client,
        judge=judge,
        output_root=tmp_path,
        timestamp="judge-ok",
        rounds=2,
        first_grid=(2, 2),
        later_grid=(2, 2),
        crop_padding=0,
        final_crop_max_size=(50, 50),
    )

    assert result.center == (240, 180)
    assert len(result.steps[0].judge_attempts) == 1
    assert result.steps[0].judge_attempts[0].contains_target is True
    assert (tmp_path / "judge-ok" / "01-judge-crop-attempt-1.png").exists()
    assert (tmp_path / "judge-ok" / "01-judge-result-attempt-1.json").exists()
    payload = json.loads((tmp_path / "judge-ok" / "01-judge-result-attempt-1.json").read_text())
    assert payload["contains_target"] is True


def test_run_llm_visual_search_revises_cell_after_judge_rejection(tmp_path):
    image = Image.new("RGB", (400, 300), "white")
    client = FakeVisionClient(
        ["R1-1-1", "R1-2-2", "R2-1-1"],
        IconDetection(
            target_visible=True,
            icon_bbox=(20, 10, 60, 50),
            confidence=0.92,
            rationale="icon is visible",
        ),
    )
    judge = FakeGridJudge([False, True, True])

    result = run_llm_visual_search(
        image,
        query="Notepad",
        client=client,
        judge=judge,
        output_root=tmp_path,
        timestamp="judge-retry",
        rounds=2,
        first_grid=(2, 2),
        later_grid=(2, 2),
        crop_padding=0,
        final_crop_max_size=(50, 50),
    )

    revise_call = [call for call in client.calls if call[0] == "revise_cell_choice"][0]
    assert revise_call[4] == ["R1-1-1"]
    assert revise_call[5] == "wrong crop"
    assert revise_call[6] == "response-1"
    assert result.steps[0].selected_cell_id == "R1-2-2"
    assert [attempt.selected_cell_id for attempt in result.steps[0].judge_attempts] == ["R1-1-1", "R1-2-2"]


def test_run_llm_visual_search_fails_after_judge_retries_are_exhausted(tmp_path):
    image = Image.new("RGB", (400, 300), "white")
    client = FakeVisionClient(
        ["R1-1-1", "R1-1-2", "R1-2-1"],
        IconDetection(
            target_visible=True,
            icon_bbox=(20, 10, 60, 50),
            confidence=0.92,
            rationale="icon is visible",
        ),
    )
    judge = FakeGridJudge([False, False, False])

    with pytest.raises(ValueError, match="Grid judge rejected all attempts"):
        run_llm_visual_search(
            image,
            query="Notepad",
            client=client,
            judge=judge,
            output_root=tmp_path,
            timestamp="judge-fail",
            rounds=1,
            first_grid=(2, 2),
            crop_padding=0,
            final_crop_max_size=(50, 50),
            max_judge_retries=2,
        )

    assert (tmp_path / "judge-fail" / "01-judge-crop-attempt-1.png").exists()
    assert (tmp_path / "judge-fail" / "01-judge-crop-attempt-2.png").exists()
    assert (tmp_path / "judge-fail" / "01-judge-crop-attempt-3.png").exists()
