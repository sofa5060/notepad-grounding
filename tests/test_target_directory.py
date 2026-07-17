from unittest.mock import MagicMock

from notepad_grounding import automation
from notepad_grounding.main import reset_target_directory


def test_save_file_creates_missing_parent_directory(monkeypatch, tmp_path):
    full_path = tmp_path / "missing" / "nested" / "post_1.txt"
    monkeypatch.setattr(automation, "pyautogui", MagicMock())
    monkeypatch.setattr(automation.time, "sleep", MagicMock())
    monkeypatch.setattr(automation, "visible_window_handles", MagicMock(return_value={1}))
    monkeypatch.setattr(automation, "wait_for_visible_window_change", MagicMock(return_value=True))

    automation.save_file(full_path=full_path)

    assert full_path.parent.is_dir()


def test_reset_target_directory_creates_missing_directory_and_clears_contents(tmp_path):
    target_dir = tmp_path / "missing" / "target"

    reset_target_directory(target_dir)
    (target_dir / "old.txt").write_text("old", encoding="utf-8")
    (target_dir / "old-folder").mkdir()
    (target_dir / "old-folder" / "old.txt").write_text("old", encoding="utf-8")

    reset_target_directory(target_dir)

    assert target_dir.is_dir()
    assert list(target_dir.iterdir()) == []
