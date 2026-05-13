from __future__ import annotations

import platform
import time
from pathlib import Path


def double_click(x: int, y: int, *, duration: float = 0.5) -> None:
    """Move mouse to (x, y) and double-click."""
    import pyautogui

    pyautogui.moveTo(x, y, duration=duration)
    pyautogui.doubleClick()


def type_text(text: str, *, interval: float = 0.01) -> None:
    """Type text at the current cursor location."""
    import pyautogui

    pyautogui.typewrite(text, interval=interval)


def press_hotkey(*keys: str) -> None:
    """Press a key combination, e.g. press_hotkey('ctrl', 'shift', 's').

    On macOS controlling a Windows VM through Parallels, Ctrl shortcuts
    must be sent as Command so Parallels translates them correctly.
    """
    import pyautogui

    mapped = list(keys)
    if platform.system() == "Darwin":
        mapped = ["command" if k == "ctrl" else k for k in mapped]

    pyautogui.hotkey(*mapped)


def sleep(seconds: float) -> None:
    """Pause execution for the given number of seconds."""
    time.sleep(seconds)


def ensure_directory(path: Path) -> Path:
    """Create the directory (and parents) if they don't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_desktop_path() -> Path:
    """Return the user's Desktop directory."""
    return Path.home() / "Desktop"


def get_target_directory() -> Path:
    """Return the target save directory: Desktop/tjm-project."""
    return get_desktop_path() / "tjm-project"


def file_exists(filename: str) -> bool:
    """Check if a file already exists in the target directory."""
    return (get_target_directory() / filename).exists()
