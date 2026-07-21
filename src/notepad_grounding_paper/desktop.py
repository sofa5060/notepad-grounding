from __future__ import annotations

import logging
import platform
import time
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def capture_desktop(*, monitor_index: int = 1) -> Image.Image:
    import mss

    with mss.MSS() as screen_capture:
        monitors = screen_capture.monitors
        if monitor_index >= len(monitors):
            raise RuntimeError(f"Monitor {monitor_index} is unavailable; found {len(monitors) - 1}.")
        screenshot = screen_capture.grab(monitors[monitor_index])

    return Image.frombytes("RGB", screenshot.size, screenshot.rgb)


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


def clear_target_directory() -> None:
    """Remove all files from the target directory to avoid replace dialogs."""
    target = get_target_directory()
    if not target.exists():
        return
    for path in target.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            import shutil
            shutil.rmtree(path)


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


def close_active_window() -> None:
    """Close the currently active window using Ctrl+W (or Cmd+W on macOS)."""
    press_hotkey("ctrl", "w")


def close_window_hard() -> None:
    """Close the active window using Alt+F4, then press Escape.

    If a wrong app is open: Alt+F4 closes it, Escape does nothing.
    If desktop is focused: Alt+F4 opens shutdown dialog, Escape DISMISSES it.
    """
    import pyautogui
    # Send Alt+F4
    pyautogui.keyDown("alt")
    pyautogui.keyDown("f4")
    pyautogui.keyUp("f4")
    pyautogui.keyUp("alt")
    sleep(0.5)
    # Send Escape to dismiss shutdown dialog if it appeared
    pyautogui.keyDown("esc")
    pyautogui.keyUp("esc")
    sleep(0.5)


def handle_recovery(action_needed: str) -> None:
    action_lower = action_needed.lower()

    if "close" in action_lower and "window" in action_lower:
        logger.info("[RECOVER] Closing active window")
        close_active_window()
        sleep(1.0)
    elif "click" in action_lower and "replace" in action_lower:
        logger.info("[RECOVER] Clicking Replace button (Enter)")
        press_hotkey("return")
        sleep(1.0)
    elif "click" in action_lower and "yes" in action_lower:
        logger.info("[RECOVER] Clicking Yes button (Enter)")
        press_hotkey("return")
        sleep(1.0)
    elif "wait" in action_lower:
        logger.info("[RECOVER] Waiting longer")
        sleep(2.0)
    else:
        logger.info("[RECOVER] Generic recovery: %s", action_needed)
        sleep(1.0)


def save_post_file(*, full_path: Path) -> None:
    press_hotkey("ctrl", "shift", "s")
    sleep(1.5)
    type_text(str(full_path), interval=0.01)
    sleep(0.5)
    press_hotkey("return")
    sleep(1.0)


def close_notepad() -> None:
    if not is_window_active("Notepad"):
        active = get_active_window_title()
        raise RuntimeError(f"Notepad not active before close. Current: {active!r}")
    press_hotkey("ctrl", "shift", "w")
    sleep(1.0)
