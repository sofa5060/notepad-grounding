from __future__ import annotations

from pathlib import Path

from direct_llm_grounding.simple_notepad_flow import filename_for_post
from direct_llm_grounding.simple_notepad_flow import format_post_content
from direct_llm_grounding.simple_notepad_flow import reset_target_directory
from direct_llm_grounding.simple_notepad_flow import target_file_for_post
from direct_llm_grounding.simple_notepad_flow import wait_for_visible_window_change


def test_format_post_content_matches_notepad_text() -> None:
    post = {"id": 7, "title": "A small title", "body": "Line one\nLine two"}

    assert format_post_content(post) == "Title: A small title\n\nLine one\nLine two"


def test_filename_for_post_uses_post_id() -> None:
    assert filename_for_post({"id": 12}, index=3) == "post_12.txt"


def test_filename_for_post_falls_back_to_index() -> None:
    assert filename_for_post({}, index=3) == "post_3.txt"


def test_target_file_for_post_joins_target_directory() -> None:
    assert target_file_for_post(Path("C:/Users/test/Desktop/tjm-project"), {"id": 4}, index=1) == Path(
        "C:/Users/test/Desktop/tjm-project/post_4.txt"
    )


def test_reset_target_directory_clears_existing_notes(tmp_path) -> None:
    (tmp_path / "old-note.txt").write_text("old", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "old-nested-note.txt").write_text("old", encoding="utf-8")

    reset_target_directory(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_wait_for_visible_window_change_returns_when_snapshot_changes() -> None:
    snapshots = iter([{1, 2}, {1, 2}, {1, 2, 3}])
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    changed = wait_for_visible_window_change(
        {1, 2},
        timeout_seconds=1.0,
        poll_seconds=0.1,
        settle_seconds=0.2,
        snapshot_fn=lambda: next(snapshots),
        sleep_fn=sleep,
        monotonic_fn=lambda: now[0],
    )

    assert changed is True
    assert sleeps == [0.1, 0.1, 0.2]


def test_wait_for_visible_window_change_times_out_without_change() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    changed = wait_for_visible_window_change(
        {1, 2},
        timeout_seconds=0.3,
        poll_seconds=0.1,
        settle_seconds=0.2,
        snapshot_fn=lambda: {1, 2},
        sleep_fn=sleep,
        monotonic_fn=lambda: now[0],
    )

    assert changed is False
    assert sleeps == [0.1, 0.1, 0.1]
