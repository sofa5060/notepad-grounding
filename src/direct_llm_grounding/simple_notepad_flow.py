from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

import pyautogui

from direct_llm_grounding.notepad_direct import ask_llm_for_coordinates
from direct_llm_grounding.notepad_direct import capture_desktop
from direct_llm_grounding.notepad_direct import draw_debug_box
from direct_llm_grounding.notepad_direct import load_env_file
from direct_llm_grounding.notepad_direct import parse_coordinate_guess
from notepad_grounding.api import fetch_posts

QUERY = "Notepad"
RUNS = 10
DELAY_BETWEEN_RUNS_SECONDS = 5
WINDOW_OPEN_TIMEOUT_SECONDS = 8.0
WINDOW_OPEN_POLL_SECONDS = 0.25
WINDOW_OPEN_SETTLE_SECONDS = 0.75
TEXT_TYPE_INTERVAL_SECONDS = 0.01
PATH_TYPE_INTERVAL_SECONDS = 0.01
OUTPUT_DIR = Path("output/simple_notepad_flow")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run() -> None:
    load_env_file()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Add it to .env or set it in the shell.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = get_target_directory()
    logger.info("clearing notes folder: %s", target_dir)
    reset_target_directory(target_dir)

    posts = fetch_posts(limit=RUNS)
    for index, post in enumerate(posts, start=1):
        filename = filename_for_post(post, index=index)
        full_path = target_file_for_post(target_dir, post, index=index)
        logger.info("[%d/%d] locating %s", index, len(posts), QUERY)

        screenshot = capture_desktop()
        screenshot_path = OUTPUT_DIR / f"{index:02d}-screenshot.png"
        screenshot.save(screenshot_path)

        raw_response = ask_llm_for_coordinates(screenshot)
        (OUTPUT_DIR / f"{index:02d}-response.txt").write_text(raw_response, encoding="utf-8")

        guess = parse_coordinate_guess(raw_response, image_size=screenshot.size)
        draw_debug_box(screenshot, guess).save(OUTPUT_DIR / f"{index:02d}-annotated.png")
        logger.info("[%s] double-clicking %s at (%d, %d)", filename, QUERY, guess.x, guess.y)

        open_notepad_at(guess.x, guess.y)
        paste_text(format_post_content(post))
        save_notepad_as(full_path)
        close_notepad()

        logger.info("[%s] saved to %s", filename, full_path)
        if index < len(posts):
            logger.info("waiting %d seconds before next run", DELAY_BETWEEN_RUNS_SECONDS)
            time.sleep(DELAY_BETWEEN_RUNS_SECONDS)


def format_post_content(post: dict) -> str:
    title = str(post.get("title", "")).strip()
    body = str(post.get("body", "")).strip()
    return f"Title: {title}\n\n{body}"


def filename_for_post(post: dict, *, index: int) -> str:
    post_id = post.get("id", index)
    return f"post_{post_id}.txt"


def target_file_for_post(target_dir: Path, post: dict, *, index: int) -> Path:
    return target_dir / filename_for_post(post, index=index)


def get_target_directory() -> Path:
    return Path.home() / "Desktop" / "tjm-project"


def reset_target_directory(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in target_dir.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def open_notepad_at(x: int, y: int) -> None:
    before_windows = visible_window_handles()
    pyautogui.moveTo(x, y, duration=0.3)
    pyautogui.doubleClick()
    if before_windows is None:
        logger.info("window snapshot unavailable; waiting %.1fs after open", WINDOW_OPEN_TIMEOUT_SECONDS)
        time.sleep(WINDOW_OPEN_TIMEOUT_SECONDS)
        return

    logger.info("waiting for visible window set to change")
    changed = wait_for_visible_window_change(
        before_windows,
        timeout_seconds=WINDOW_OPEN_TIMEOUT_SECONDS,
        poll_seconds=WINDOW_OPEN_POLL_SECONDS,
        settle_seconds=WINDOW_OPEN_SETTLE_SECONDS,
    )
    if not changed:
        logger.warning("visible window set did not change within %.1fs; continuing", WINDOW_OPEN_TIMEOUT_SECONDS)


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


def wait_for_visible_window_change(
    before_windows: set[int],
    *,
    timeout_seconds: float = WINDOW_OPEN_TIMEOUT_SECONDS,
    poll_seconds: float = WINDOW_OPEN_POLL_SECONDS,
    settle_seconds: float = WINDOW_OPEN_SETTLE_SECONDS,
    snapshot_fn=visible_window_handles,
    sleep_fn=time.sleep,
    monotonic_fn=time.monotonic,
) -> bool:
    deadline = monotonic_fn() + timeout_seconds
    while monotonic_fn() < deadline:
        current_windows = snapshot_fn()
        if current_windows is not None and current_windows != before_windows:
            sleep_fn(settle_seconds)
            return True
        sleep_fn(poll_seconds)
    return False


def paste_text(text: str) -> None:
    logger.info("typing %d characters into Notepad", len(text))
    pyautogui.write(text, interval=TEXT_TYPE_INTERVAL_SECONDS)
    time.sleep(0.5)


def save_notepad_as(full_path: Path) -> None:
    pyautogui.hotkey("ctrl", "shift", "s")
    time.sleep(1.5)
    pyautogui.hotkey("alt", "n")
    time.sleep(0.2)
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.2)
    logger.info("typing save path: %s", full_path)
    pyautogui.write(str(full_path), interval=PATH_TYPE_INTERVAL_SECONDS)
    time.sleep(0.5)
    pyautogui.press("enter")
    time.sleep(1.0)


def close_notepad() -> None:
    pyautogui.hotkey("alt", "f4")
    time.sleep(1.0)


if __name__ == "__main__":
    run()
