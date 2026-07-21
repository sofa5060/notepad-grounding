from dataclasses import dataclass
from unittest.mock import MagicMock
from unittest.mock import patch

from notepad_grounding_paper.flow import FlowDependencies
from notepad_grounding_paper.flow import run_flow
from notepad_grounding_paper.models import DesktopReviewResult


class FakeDesktopReviewer:
    """Always returns success so tests don't need real OpenAI calls."""

    def review_desktop_state(self, *, action, expected, image):
        return DesktopReviewResult(
            status="success",
            action_needed="proceed",
            rationale="fake reviewer says all good",
        )


def fake_dependencies() -> FlowDependencies:
    return FlowDependencies(
        client=object(),
        reviewer=FakeDesktopReviewer(),
    )


@dataclass(frozen=True)
class FakeLocateResult:
    center: tuple[int, int] = (100, 100)
    output_dir: str = "locate-output"
    elapsed_seconds: float = 0.0


def test_run_flow_locate_mode_captures_once_and_calls_locator(tmp_path):
    mock_image = MagicMock()

    with (
        patch("notepad_grounding_paper.flow.build_default_dependencies", return_value=fake_dependencies()),
        patch("notepad_grounding_paper.desktop.capture_desktop", return_value=mock_image) as mock_capture,
        patch("notepad_grounding_paper.flow.run_locate", return_value=FakeLocateResult()) as mock_locate,
    ):
        result = run_flow(
            mode="locate",
            query="Notepad",
            output_root=tmp_path,
            timestamp="test",
        )

    assert result.center == (100, 100)
    mock_capture.assert_called_once()
    mock_locate.assert_called_once()
    assert mock_locate.call_args.kwargs["query"] == "Notepad"
    assert mock_locate.call_args.kwargs["output_root"] == tmp_path / "locate"


def test_run_flow_run_mode_processes_posts_and_actions(tmp_path):
    posts = [
        {"id": 1, "title": "Post One", "body": "Body one"},
        {"id": 2, "title": "Post Two", "body": "Body two"},
    ]
    mock_image = MagicMock()
    mock_image.width = 1920
    mock_image.height = 1080

    with (
        patch("notepad_grounding_paper.flow.build_default_dependencies", return_value=fake_dependencies()),
        patch("notepad_grounding_paper.flow.fetch_posts", return_value=posts),
        patch("notepad_grounding_paper.desktop.capture_desktop", return_value=mock_image),
        patch("notepad_grounding_paper.flow.run_locate", return_value=FakeLocateResult()),
        patch("notepad_grounding_paper.desktop.prepare_target_directory", return_value=tmp_path) as mock_prepare,
        patch("notepad_grounding_paper.desktop.double_click") as mock_click,
        patch("notepad_grounding_paper.desktop.type_text") as mock_type,
        patch("notepad_grounding_paper.desktop.press_hotkey") as mock_hotkey,
        patch("notepad_grounding_paper.desktop.sleep"),
        patch("notepad_grounding_paper.desktop.pyautogui.getActiveWindowTitle", create=True, return_value="Untitled - Notepad"),
    ):
        result = run_flow(
            mode="run",
            query="Notepad",
            output_root=tmp_path,
            timestamp="test",
            max_retries=1,
            post_limit=2,
        )

    assert result.total_posts == 2
    assert result.succeeded == 2
    assert result.failed == 0
    mock_prepare.assert_called_once()
    assert mock_click.call_count == 2
    mock_type.assert_called()
    mock_hotkey.assert_called()


def test_run_flow_retries_failed_post_attempts(tmp_path):
    posts = [{"id": 1, "title": "Post One", "body": "Body one"}]
    mock_image = MagicMock()
    call_count = 0

    def flaky_locate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("Grounding failed")
        return FakeLocateResult()

    with (
        patch("notepad_grounding_paper.flow.build_default_dependencies", return_value=fake_dependencies()),
        patch("notepad_grounding_paper.flow.fetch_posts", return_value=posts),
        patch("notepad_grounding_paper.desktop.capture_desktop", return_value=mock_image),
        patch("notepad_grounding_paper.flow.run_locate", side_effect=flaky_locate),
        patch("notepad_grounding_paper.desktop.prepare_target_directory", return_value=tmp_path),
        patch("notepad_grounding_paper.desktop.double_click") as mock_click,
        patch("notepad_grounding_paper.desktop.type_text") as mock_type,
        patch("notepad_grounding_paper.desktop.press_hotkey") as mock_hotkey,
        patch("notepad_grounding_paper.desktop.sleep"),
        patch("notepad_grounding_paper.desktop.pyautogui.getActiveWindowTitle", create=True, return_value="Untitled - Notepad"),
    ):
        result = run_flow(
            mode="run",
            query="Notepad",
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


def test_run_flow_records_failure_after_max_retries(tmp_path):
    posts = [{"id": 1, "title": "Post One", "body": "Body one"}]
    mock_image = MagicMock()

    with (
        patch("notepad_grounding_paper.flow.build_default_dependencies", return_value=fake_dependencies()),
        patch("notepad_grounding_paper.flow.fetch_posts", return_value=posts),
        patch("notepad_grounding_paper.desktop.capture_desktop", return_value=mock_image),
        patch("notepad_grounding_paper.flow.run_locate", side_effect=RuntimeError("Always fails")),
        patch("notepad_grounding_paper.desktop.prepare_target_directory", return_value=tmp_path),
        patch("notepad_grounding_paper.desktop.double_click") as mock_click,
        patch("notepad_grounding_paper.desktop.type_text") as mock_type,
        patch("notepad_grounding_paper.desktop.press_hotkey") as mock_hotkey,
        patch("notepad_grounding_paper.desktop.sleep"),
    ):
        result = run_flow(
            mode="run",
            query="Notepad",
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
