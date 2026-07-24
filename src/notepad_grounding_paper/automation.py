from __future__ import annotations

import logging
import platform
import time
from pathlib import Path

import pyautogui

from notepad_grounding_paper import llm

logger = logging.getLogger(__name__)


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


def close_window_hard() -> None:
    pyautogui.hotkey("alt", "f4")
    time.sleep(0.5)
    # Escape dismisses the shutdown dialog in case Alt+F4 hit the bare desktop.
    pyautogui.press("esc")
    time.sleep(0.5)


def handle_recovery(action_needed: str) -> None:
    logger.info("[RECOVER] %s", action_needed)
    action = action_needed.lower()
    if "close" in action and "window" in action:
        press_hotkey("ctrl", "w")
        time.sleep(1.0)
    elif "click" in action and ("replace" in action or "yes" in action):
        press_hotkey("return")
        time.sleep(1.0)
    elif "wait" in action:
        time.sleep(2.0)
    else:
        time.sleep(1.0)


def open_app(*, query: str, center: tuple[int, int]) -> None:
    """Double-click the located icon, then verify Notepad actually opened."""
    double_click(*center)
    time.sleep(2.0)
    review = llm.review_desktop_state(
        action=f"Double-clicked the {query!r} desktop icon at {center}",
        expected="Notepad window is open",
        image=llm.capture_desktop(),
    )
    if review.status == "wrong_app":
        logger.warning("reviewer detected wrong app: %s", review.rationale)
        close_window_hard()
        time.sleep(2.0)
        raise RuntimeError(f"Wrong app opened: {review.rationale}")
    if review.status in ("error", "retry"):
        handle_recovery(review.action_needed)
        raise RuntimeError(f"Open failed: {review.rationale}")


def type_post(*, title: str, body: str) -> None:
    """Type the post into Notepad and sanity-check the screen afterwards."""
    type_text(f"Title: {title}\n\n{body}")
    time.sleep(0.5)
    review = llm.review_desktop_state(
        action="Typed post content into Notepad",
        expected="Notepad shows the typed post text",
        image=llm.capture_desktop(),
    )
    if review.status != "success":
        logger.warning("reviewer detected typing issue: %s", review.rationale)


def save_and_close(*, full_path: Path) -> None:
    """Save the file, clear any pop-up dialogs, then close Notepad."""
    press_hotkey("ctrl", "shift", "s")
    time.sleep(1.5)
    type_text(str(full_path))
    time.sleep(0.5)
    press_hotkey("return")
    time.sleep(1.0)

    # Up to 3 review cycles to clear pop-ups (e.g. an overwrite confirmation).
    for _ in range(3):
        review = llm.review_desktop_state(
            action=f"Pressed Save with path {full_path}",
            expected="File is saved, no dialogs remain, Notepad is open",
            image=llm.capture_desktop(),
        )
        if review.status == "success":
            break
        if review.status == "pop_up":
            logger.info("reviewer detected pop-up: %s", review.rationale)
            handle_recovery(review.action_needed)
            time.sleep(1.0)
            continue
        if review.status == "wrong_app":
            logger.warning("reviewer detected wrong window: %s", review.rationale)
            handle_recovery(review.action_needed)
            raise RuntimeError(f"Save went wrong: {review.rationale}")
        handle_recovery(review.action_needed)
        break
    else:
        raise RuntimeError("Pop-up handling exceeded max cycles")

    close_notepad()
    review = llm.review_desktop_state(
        action="Closed Notepad", expected="Notepad is closed, desktop is visible", image=llm.capture_desktop()
    )
    if review.status != "success":
        logger.warning("reviewer detected close issue: %s", review.rationale)
        handle_recovery(review.action_needed)


def close_notepad() -> None:
    try:
        active = pyautogui.getActiveWindowTitle()
    except Exception:
        active = None
    if not active or "notepad" not in active.lower():
        raise RuntimeError(f"Notepad not active before close. Current: {active!r}")
    press_hotkey("ctrl", "shift", "w")
    time.sleep(1.0)


def automate_post(*, post: dict, query: str, center: tuple[int, int], target_dir: Path) -> Path:
    """Write one post into Notepad: open at the located icon, type, save, close."""
    full_path = target_dir / f"post_{post['id']}.txt"
    open_app(query=query, center=center)
    type_post(title=post["title"], body=post["body"])
    save_and_close(full_path=full_path)
    return full_path
