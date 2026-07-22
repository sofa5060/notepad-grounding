from __future__ import annotations

import logging
import platform
import shutil
import time
from pathlib import Path

import mss
import pyautogui
from PIL import Image

logger = logging.getLogger(__name__)

sleep = time.sleep


def capture_desktop() -> Image.Image:
    with mss.MSS() as screen_capture:
        screenshot = screen_capture.grab(screen_capture.monitors[1])
    return Image.frombytes("RGB", screenshot.size, screenshot.rgb)


def double_click(x: int, y: int, *, duration: float = 0.5) -> None:
    pyautogui.moveTo(x, y, duration=duration)
    pyautogui.doubleClick()


def type_text(text: str, *, interval: float = 0.01) -> None:
    pyautogui.typewrite(text, interval=interval)


def press_hotkey(*keys: str) -> None:
    # Parallels needs Cmd instead of Ctrl when driving the VM from macOS.
    if platform.system() == "Darwin":
        keys = tuple("command" if key == "ctrl" else key for key in keys)
    pyautogui.hotkey(*keys)


def prepare_target_directory() -> Path:
    target = Path.home() / "Desktop" / "tjm-project"
    target.mkdir(parents=True, exist_ok=True)
    for path in target.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    return target


def close_window_hard() -> None:
    pyautogui.hotkey("alt", "f4")
    sleep(0.5)
    # Escape dismisses the shutdown dialog in case Alt+F4 hit the bare desktop.
    pyautogui.press("esc")
    sleep(0.5)


def handle_recovery(action_needed: str) -> None:
    logger.info("[RECOVER] %s", action_needed)
    action = action_needed.lower()
    if "close" in action and "window" in action:
        press_hotkey("ctrl", "w")
        sleep(1.0)
    elif "click" in action and ("replace" in action or "yes" in action):
        press_hotkey("return")
        sleep(1.0)
    elif "wait" in action:
        sleep(2.0)
    else:
        sleep(1.0)


def save_post_file(*, full_path: Path) -> None:
    press_hotkey("ctrl", "shift", "s")
    sleep(1.5)
    type_text(str(full_path))
    sleep(0.5)
    press_hotkey("return")
    sleep(1.0)


def close_notepad() -> None:
    try:
        active = pyautogui.getActiveWindowTitle()
    except Exception:
        active = None
    if not active or "notepad" not in active.lower():
        raise RuntimeError(f"Notepad not active before close. Current: {active!r}")
    press_hotkey("ctrl", "shift", "w")
    sleep(1.0)
