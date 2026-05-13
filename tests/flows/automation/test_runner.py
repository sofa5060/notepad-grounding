from dataclasses import dataclass
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from notepad_grounding.flows.automation.runner import PostResult
from notepad_grounding.flows.automation.runner import run_automation
from notepad_grounding.shared.llm import CellChoice
from notepad_grounding.shared.llm import CellsChoice


class FakeVisionClient:
    def choose_cell(self, *, query, image, cell_ids):
        return CellChoice(cell_id=cell_ids[0], confidence=0.9, rationale="test")

    def choose_cells(self, *, query, image, cell_ids):
        return CellsChoice(cell_ids=cell_ids[:1], confidence=0.9, rationale="test")


def test_run_automation_overwrites_existing_files(tmp_path):
    posts = [
        {"id": 1, "title": "Post One", "body": "Body one"},
        {"id": 2, "title": "Post Two", "body": "Body two"},
    ]

    @dataclass(frozen=True)
    class FakeLocateResult:
        center: tuple[int, int] = (100, 100)
        output_dir: str = str(tmp_path)

    with (
        patch("notepad_grounding.flows.automation.runner.fetch_posts", return_value=posts),
        patch("notepad_grounding.flows.automation.runner.capture_desktop") as mock_capture,
        patch("notepad_grounding.flows.automation.runner.double_click") as mock_click,
        patch("notepad_grounding.flows.automation.runner.type_text") as mock_type,
        patch("notepad_grounding.flows.automation.runner.press_hotkey") as mock_hotkey,
        patch("notepad_grounding.flows.automation.runner.sleep"),
        patch("notepad_grounding.flows.automation.runner.ensure_directory"),
        patch("notepad_grounding.flows.automation.runner.get_target_directory", return_value=tmp_path),
        patch("notepad_grounding.flows.automation.runner.run_llm_visual_search", return_value=FakeLocateResult()),
    ):
        mock_image = MagicMock()
        mock_image.width = 1920
        mock_image.height = 1080
        mock_capture.return_value = mock_image

        client = FakeVisionClient()
        result = run_automation(
            query="Notepad",
            client=client,
            output_root=tmp_path,
            timestamp="test",
            max_retries=1,
            post_limit=2,
        )

    assert result.total_posts == 2
    assert result.succeeded == 2
    assert result.failed == 0
    assert mock_click.call_count == 2
    mock_type.assert_called()
    mock_hotkey.assert_called()


def test_run_automation_retries_on_failure(tmp_path):
    posts = [{"id": 1, "title": "Post One", "body": "Body one"}]

    with (
        patch("notepad_grounding.flows.automation.runner.fetch_posts", return_value=posts),
        patch("notepad_grounding.flows.automation.runner.capture_desktop") as mock_capture,
        patch("notepad_grounding.flows.automation.runner.double_click") as mock_click,
        patch("notepad_grounding.flows.automation.runner.type_text") as mock_type,
        patch("notepad_grounding.flows.automation.runner.press_hotkey") as mock_hotkey,
        patch("notepad_grounding.flows.automation.runner.sleep"),
        patch("notepad_grounding.flows.automation.runner.ensure_directory"),
        patch("notepad_grounding.flows.automation.runner.get_target_directory", return_value=tmp_path),
    ):
        mock_image = MagicMock()
        mock_image.width = 1920
        mock_image.height = 1080
        mock_capture.return_value = mock_image

        # Fail twice, succeed on third attempt
        call_count = 0

        def flaky_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Grounding failed")
            # Return minimal result
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class FakeResult:
                center: tuple[int, int] = (100, 100)
                output_dir: str = str(tmp_path)

            return FakeResult()

        with patch("notepad_grounding.flows.automation.runner.run_llm_visual_search", side_effect=flaky_llm):
            client = FakeVisionClient()
            result = run_automation(
                query="Notepad",
                client=client,
                output_root=tmp_path,
                timestamp="test",
                max_retries=3,
                retry_delay=0.0,
                post_limit=1,
            )

    assert result.total_posts == 1
    assert result.succeeded == 1
    assert result.failed == 0
    assert call_count == 3
    mock_click.assert_called_once()
    mock_type.assert_called()
    mock_hotkey.assert_called()


def test_run_automation_fails_after_max_retries(tmp_path):
    posts = [{"id": 1, "title": "Post One", "body": "Body one"}]

    with (
        patch("notepad_grounding.flows.automation.runner.fetch_posts", return_value=posts),
        patch("notepad_grounding.flows.automation.runner.capture_desktop") as mock_capture,
        patch("notepad_grounding.flows.automation.runner.double_click") as mock_click,
        patch("notepad_grounding.flows.automation.runner.type_text") as mock_type,
        patch("notepad_grounding.flows.automation.runner.press_hotkey") as mock_hotkey,
        patch("notepad_grounding.flows.automation.runner.sleep"),
        patch("notepad_grounding.flows.automation.runner.ensure_directory"),
        patch("notepad_grounding.flows.automation.runner.get_target_directory", return_value=tmp_path),
    ):
        mock_image = MagicMock()
        mock_image.width = 1920
        mock_image.height = 1080
        mock_capture.return_value = mock_image

        def always_fail(*args, **kwargs):
            raise RuntimeError("Always fails")

        with patch("notepad_grounding.flows.automation.runner.run_llm_visual_search", side_effect=always_fail):
            client = FakeVisionClient()
            result = run_automation(
                query="Notepad",
                client=client,
                output_root=tmp_path,
                timestamp="test",
                max_retries=2,
                retry_delay=0.0,
                post_limit=1,
            )

    assert result.total_posts == 1
    assert result.succeeded == 0
    assert result.failed == 1
    mock_click.assert_not_called()
    mock_type.assert_not_called()
    mock_hotkey.assert_not_called()
