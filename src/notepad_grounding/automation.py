from __future__ import annotations

import logging
import os
import shutil
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


def reset_target_directory(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in target_dir.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def format_post_content(post: dict) -> str:
    title = str(post.get("title", "")).strip()
    body = str(post.get("body", "")).strip()
    return f"Title: {title}\n\n{body}"


def target_file_for_post(target_dir: Path, post: dict, *, index: int) -> Path:
    post_id = post.get("id", index)
    return target_dir / f"post_{post_id}.txt"


def open_icon(x: int, y: int) -> None:
    before_windows = visible_window_handles()
    pyautogui.moveTo(x, y, duration=0.3)
    pyautogui.doubleClick()
    wait_for_window_change_or_sleep(before_windows, action="open", timeout_seconds=WINDOW_OPEN_TIMEOUT_SECONDS)


def type_post(post: dict) -> None:
    text = format_post_content(post)
    logger.info("typing %d characters into Notepad", len(text))
    pyautogui.write(text, interval=TEXT_TYPE_INTERVAL_SECONDS)
    time.sleep(0.5)


def save_as(full_path: Path) -> None:
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
    wait_for_window_change_or_sleep(before_windows, action="save", timeout_seconds=WINDOW_CLOSE_TIMEOUT_SECONDS)


def close_notepad() -> None:
    before_windows = visible_window_handles()
    logger.info("closing Notepad with Ctrl+Shift+W")
    pyautogui.hotkey("ctrl", "shift", "w")
    if wait_for_window_change_or_sleep(before_windows, action="close", timeout_seconds=WINDOW_CLOSE_TIMEOUT_SECONDS):
        return

    logger.warning("Ctrl+Shift+W did not close Notepad; trying Alt+F4")
    before_alt_f4_windows = visible_window_handles()
    press_alt_f4()
    wait_for_window_change_or_sleep(before_alt_f4_windows, action="Alt+F4 close", timeout_seconds=WINDOW_CLOSE_TIMEOUT_SECONDS)


def wait_for_window_change_or_sleep(before_windows: set[int] | None, *, action: str, timeout_seconds: float) -> bool:
    if before_windows is None:
        time.sleep(min(timeout_seconds, 1.0))
        return False

    logger.info("waiting for visible window set to change after %s", action)
    changed = wait_for_visible_window_change(before_windows, timeout_seconds=timeout_seconds)
    if not changed:
        logger.warning("visible window set did not change within %.1fs after %s", timeout_seconds, action)
    return changed


def visible_window_handles() -> set[int] | None:
    if os.name != "nt":
        return None

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    handles: set[int] = set()

    def enum_callback(hwnd: int, _lparam: int) -> bool:
        if user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd) > 0:
            handles.add(hwnd)
        return True

    enum_windows_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(enum_windows_proc(enum_callback), 0)
    return handles


def wait_for_visible_window_change(before_windows: set[int], *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        current_windows = visible_window_handles()
        if current_windows is not None and current_windows != before_windows:
            time.sleep(WINDOW_SETTLE_SECONDS)
            return True
        time.sleep(WINDOW_POLL_SECONDS)
    return False


def press_alt_f4() -> None:
    pyautogui.keyDown("alt")
    pyautogui.keyDown("f4")
    pyautogui.keyUp("f4")
    pyautogui.keyUp("alt")
