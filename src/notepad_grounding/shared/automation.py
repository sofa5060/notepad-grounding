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


def get_active_window_title() -> str | None:
    """Return the title of the currently active window."""
    try:
        import pyautogui
        return pyautogui.getActiveWindowTitle()
    except Exception:
        return None


def is_window_active(title_substring: str) -> bool:
    """Check if the active window title contains the given substring."""
    active = get_active_window_title()
    if active is None:
        return False
    return title_substring.lower() in active.lower()


def wait_for_window(title_substring: str, *, timeout: float = 5.0, poll_interval: float = 0.5) -> bool:
    """Wait until a window with the given title substring becomes active.

    Returns True if found, False if timeout exceeded.
    """
    import pyautogui

    elapsed = 0.0
    while elapsed < timeout:
        active = get_active_window_title()
        if active and title_substring.lower() in active.lower():
            return True
        time.sleep(poll_interval)
        elapsed += poll_interval
    return False


def wait_for_window_close(title_substring: str, *, timeout: float = 5.0, poll_interval: float = 0.5) -> bool:
    """Wait until a window with the given title substring is no longer active.

    Returns True if closed, False if timeout exceeded.
    """
    elapsed = 0.0
    while elapsed < timeout:
        active = get_active_window_title()
        if active is None or title_substring.lower() not in active.lower():
            return True
        time.sleep(poll_interval)
        elapsed += poll_interval
    return False


def close_active_window() -> None:
    """Close the currently active window using Ctrl+W (or Cmd+W on macOS)."""
    press_hotkey("ctrl", "w")


def click_at(x: int, y: int, *, clicks: int = 1, duration: float = 0.5) -> None:
    """Click at screen coordinates."""
    import pyautogui
    pyautogui.moveTo(x, y, duration=duration)
    if clicks == 2:
        pyautogui.doubleClick()
    else:
        pyautogui.click()
