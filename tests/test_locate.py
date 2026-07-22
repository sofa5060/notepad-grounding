import pytest
from PIL import Image

from notepad_grounding_paper.locate import run_locate
from notepad_grounding_paper.models import CellChoice
from notepad_grounding_paper.models import TargetReviewResult


class FakeVisionClient:
    def __init__(self, choices, grid_choices=None, fail_points=False):
        self.choices = list(choices)
        self.grid_choices = list(grid_choices or [])
        self.fail_points = fail_points
        self.calls = []

    def choose_cell(self, *, query, image, cell_ids):
        self.calls.append(("choose_cell", query, image.size, cell_ids))
        index = len([call for call in self.calls if call[0] in {"choose_cell", "revise_cell_choice"}])
        return CellChoice(
            cell_id=self.choices.pop(0), confidence=0.9, rationale="test choice", response_id=f"response-{index}"
        )

    def revise_cell_choice(
        self, *, query, image, cell_ids, rejected_cell_ids, reviewer_rationale, previous_response_id
    ):
        self.calls.append(
            (
                "revise_cell_choice",
                query,
                image.size,
                cell_ids,
                list(rejected_cell_ids),
                reviewer_rationale,
                previous_response_id,
            )
        )
        index = len([call for call in self.calls if call[0] in {"choose_cell", "revise_cell_choice"}])
        return CellChoice(
            cell_id=self.choices.pop(0), confidence=0.8, rationale="revised choice", response_id=f"response-{index}"
        )

    def choose_click_grid_cell(self, *, query, image, cell_ids, rejected_cell_ids=None, previous_response_id=None):
        self.calls.append(
            ("choose_click_grid_cell", query, image.size, cell_ids, list(rejected_cell_ids or []), previous_response_id)
        )
        if self.fail_points:
            raise ValueError("grid selection failed")
        return CellChoice(
            cell_id=self.grid_choices.pop(0),
            confidence=0.88,
            rationale="test grid cell",
            response_id=f"grid-response-{len(self.calls)}",
        )


class FakeTargetReviewer:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = []

    def review_target_crop(self, *, query, image):
        self.calls.append((query, image.size))
        contains_target = self.verdicts.pop(0)
        return TargetReviewResult(
            contains_target=contains_target,
            confidence=0.95 if contains_target else 0.85,
            rationale="accepted" if contains_target else "wrong crop",
            visible_evidence="notepad visible" if contains_target else "notepad absent",
        )

    def review_target_grid_cell(self, *, query, image):
        return self.review_target_crop(query=query, image=image)


def test_run_locate_maps_selected_cells_to_screen_coordinates(tmp_path):
    image = Image.new("RGB", (400, 300), "white")
    client = FakeVisionClient(["R1-2-2", "R2-1-1"], grid_choices=["R4C4", "R3C3"])

    result = run_locate(
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

    assert result.center == (250, 187)
    assert client.calls[0][1] == "Notepad"
    assert client.calls[0][3] == ["R1-1-1", "R1-1-2", "R1-2-1", "R1-2-2"]
    for name in [
        "00-source.png",
        "01-grid.png",
        "01-selected.png",
        "02-selected.png",
        "final-crop.png",
        "click-points-01.png",
        "click-points-02-crop.png",
        "click-points-02.png",
        "click-point-final.png",
        "click-point-full.png",
    ]:
        assert (tmp_path / "run" / name).exists()


def test_run_locate_with_review_accepts_first_choice_and_saves_crops(tmp_path):
    image = Image.new("RGB", (400, 300), "white")
    client = FakeVisionClient(["R1-2-2", "R2-1-1"], grid_choices=["R4C4", "R3C3"])
    target_reviewer = FakeTargetReviewer([True, True, True, True])

    result = run_locate(
        image,
        query="Notepad",
        client=client,
        target_reviewer=target_reviewer,
        output_root=tmp_path,
        timestamp="review-ok",
        rounds=2,
        first_grid=(2, 2),
        later_grid=(2, 2),
        crop_padding=0,
        final_crop_max_size=(50, 50),
    )

    assert result.center == (250, 187)
    assert len(target_reviewer.calls) == 4  # 2 zoom rounds + 2 click-grid passes
    assert all(call[0] != "revise_cell_choice" for call in client.calls)
    assert (tmp_path / "review-ok" / "01-target-review-crop-attempt-1.png").exists()


def test_run_locate_revises_cell_after_review_rejection(tmp_path):
    image = Image.new("RGB", (400, 300), "white")
    client = FakeVisionClient(["R1-1-1", "R1-2-2", "R2-1-1"], grid_choices=["R4C4", "R3C3"])
    target_reviewer = FakeTargetReviewer([False, True, True, True, True])

    result = run_locate(
        image,
        query="Notepad",
        client=client,
        target_reviewer=target_reviewer,
        output_root=tmp_path,
        timestamp="review-retry",
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
    assert result.center == (250, 187)
    assert (tmp_path / "review-retry" / "01-target-review-crop-attempt-2.png").exists()


def test_run_locate_fails_after_review_retries_are_exhausted(tmp_path):
    image = Image.new("RGB", (400, 300), "white")
    client = FakeVisionClient(["R1-1-1", "R1-1-2", "R1-2-1"])
    target_reviewer = FakeTargetReviewer([False, False, False])

    with pytest.raises(ValueError, match="Target reviewer rejected all attempts"):
        run_locate(
            image,
            query="Notepad",
            client=client,
            target_reviewer=target_reviewer,
            output_root=tmp_path,
            timestamp="review-fail",
            rounds=1,
            first_grid=(2, 2),
            crop_padding=0,
            final_crop_max_size=(50, 50),
            max_review_retries=2,
        )

    assert (tmp_path / "review-fail" / "01-target-review-crop-attempt-1.png").exists()
    assert (tmp_path / "review-fail" / "01-target-review-crop-attempt-2.png").exists()
    assert (tmp_path / "review-fail" / "01-target-review-crop-attempt-3.png").exists()


def test_run_locate_retries_marked_grid_after_review_rejection(tmp_path):
    image = Image.new("RGB", (400, 300), "white")
    client = FakeVisionClient(["R1-2-2", "R2-1-1"], grid_choices=["R1C1", "R4C4", "R3C3"])
    target_reviewer = FakeTargetReviewer([True, True, False, True, True])

    run_locate(
        image,
        query="Notepad",
        client=client,
        target_reviewer=target_reviewer,
        output_root=tmp_path,
        timestamp="points-review",
        rounds=2,
        first_grid=(2, 2),
        later_grid=(2, 2),
        crop_padding=0,
        final_crop_max_size=(50, 50),
    )

    grid_calls = [call for call in client.calls if call[0] == "choose_click_grid_cell"]
    assert grid_calls[1][4] == ["R1C1"]
    assert grid_calls[1][5] == "grid-response-3"
    assert (tmp_path / "points-review" / "click-grid-01-target-review-crop-attempt-1.png").exists()
    assert (tmp_path / "points-review" / "click-grid-01-target-review-crop-attempt-2.png").exists()


def test_run_locate_raises_when_marked_points_fail(tmp_path):
    image = Image.new("RGB", (400, 300), "white")
    client = FakeVisionClient(["R1-2-2", "R2-1-1"], fail_points=True)

    with pytest.raises(ValueError, match="grid selection failed"):
        run_locate(
            image,
            query="Notepad",
            client=client,
            output_root=tmp_path,
            timestamp="points-fail",
            rounds=2,
            first_grid=(2, 2),
            later_grid=(2, 2),
            crop_padding=0,
            final_crop_max_size=(50, 50),
        )
