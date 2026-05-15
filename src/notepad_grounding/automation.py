from __future__ import annotations
import ctypes
from ctypes import wintypes
import logging
import platform
import time
from pathlib import Path
import pyautogui

WINDOW_OPEN_TIMEOUT_SECONDS = 8.0
WINDOW_CLOSE_TIMEOUT_SECONDS = 5.0
WINDOW_POLL_SECONDS = 0.25
WINDOW_SETTLE_SECONDS = 0.75
TEXT_TYPE_INTERVAL_SECONDS = 0.01
PATH_TYPE_INTERVAL_SECONDS = 0.01

logger = logging.getLogger(__name__)


def click_icon(*, x: int, y: int) -> None:
    before_windows = visible_window_handles()
    pyautogui.moveTo(x, y, duration=0.3)
    pyautogui.doubleClick()
    wait_for_visible_window_change(before_windows, action="open", timeout_seconds=WINDOW_OPEN_TIMEOUT_SECONDS)


def type_content(*, content: str) -> None:
    logger.info("typing %d characters into Notepad", len(content))
    pyautogui.write(content, interval=TEXT_TYPE_INTERVAL_SECONDS)
    time.sleep(0.5)


def save_file(*, full_path: Path) -> None:
    pyautogui.hotkey("ctrl", "shift", "s")
    time.sleep(1.5)
    pyautogui.hotkey("alt", "n")
    time.sleep(0.2)
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.2)
    logger.info("typing save path: %s", full_path)
    pyautogui.write(str(full_path), interval=PATH_TYPE_INTERVAL_SECONDS)
    time.sleep(0.5)
    before_windows = visible_window_handles()
    pyautogui.press("enter")
    wait_for_visible_window_change(before_windows, action="save", timeout_seconds=WINDOW_CLOSE_TIMEOUT_SECONDS)


def close_notepad() -> None:
    before_windows = visible_window_handles()
    logger.info("closing Notepad with Ctrl+Shift+W")
    pyautogui.hotkey("ctrl", "shift", "w")
    if not wait_for_visible_window_change(before_windows, action="close", timeout_seconds=WINDOW_CLOSE_TIMEOUT_SECONDS):
        logger.warning("Ctrl+Shift+W did not close Notepad; trying Alt+F4")
        before_windows = visible_window_handles()
        pyautogui.keyDown("alt")
        pyautogui.keyDown("f4")
        pyautogui.keyUp("f4")
        pyautogui.keyUp("alt")
        wait_for_visible_window_change(before_windows, action="Alt+F4 close", timeout_seconds=WINDOW_CLOSE_TIMEOUT_SECONDS)


def automate_post(*, post: dict, index: int, target_dir: Path, click_x: int, click_y: int) -> Path:
    post_id = post.get("id", index)
    full_path = target_dir / f"post_{post_id}.txt"
    title = str(post.get("title", "")).strip()
    body = str(post.get("body", "")).strip()
    content = f"Title: {title}\n\n{body}"

    click_icon(x=click_x, y=click_y)
    type_content(content=content)
    save_file(full_path=full_path)
    close_notepad()

    return full_path


def visible_window_handles() -> set[int] | None:
    if platform.system() != "Windows":
        return None

    user32 = ctypes.windll.user32
    handles: set[int] = set()

    def enum_callback(hwnd: int, _lparam: int) -> bool:
        if user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd) > 0:
            handles.add(hwnd)
        return True

    enum_windows_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(enum_windows_proc(enum_callback), 0)
    return handles


def wait_for_visible_window_change(before_windows: set[int] | None, *, action: str, timeout_seconds: float) -> bool:
    if before_windows is None:
        time.sleep(min(timeout_seconds, 1.0))
        return False

    logger.info("waiting for visible window set to change after %s", action)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        current_windows = visible_window_handles()
        if current_windows is not None and current_windows != before_windows:
            time.sleep(WINDOW_SETTLE_SECONDS)
            return True
        time.sleep(WINDOW_POLL_SECONDS)

    logger.warning("visible window set did not change within %.1fs after %s", timeout_seconds, action)
    return False
