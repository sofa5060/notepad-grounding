from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from notepad_grounding.desktop import close_active_window
from notepad_grounding.desktop import close_window_hard
from notepad_grounding.desktop import double_click
from notepad_grounding.desktop import get_active_window_title
from notepad_grounding.desktop import is_window_active
from notepad_grounding.desktop import press_hotkey
from notepad_grounding.desktop import sleep
from notepad_grounding.desktop import type_text
from notepad_grounding.models import DesktopReviewResult

logger = logging.getLogger(__name__)


def review_desktop_state(
    desktop_reviewer,
    *,
    action: str,
    expected: str,
    image: Image.Image,
) -> DesktopReviewResult:
    logger.info("[REVIEW] %s | Expected: %s", action, expected)
    result = desktop_reviewer.review_desktop_state(action=action, expected=expected, image=image)
    logger.info("[REVIEW] status=%s action_needed=%s rationale=%s", result.status, result.action_needed, result.rationale)
    return result


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


def open_target_icon(*, filename: str, query: str, center: tuple[int, int]) -> None:
    double_click(*center)
    logger.info("[%s] Double-clicked the %r icon at %s", filename, query, center)
    sleep(2.0)


def type_post_content(*, filename: str, title: str, body: str) -> str:
    content = f"Title: {title}\n\n{body}"
    type_text(content)
    logger.info("[%s] Typed post content (%d chars)", filename, len(content))
    sleep(0.5)
    return content


def save_post_file(*, filename: str, full_path: Path) -> None:
    press_hotkey("ctrl", "shift", "s")
    sleep(1.5)
    type_text(str(full_path), interval=0.01)
    sleep(0.5)
    press_hotkey("return")
    sleep(1.0)
    logger.info("[%s] Save triggered", filename)


def close_notepad(*, filename: str) -> None:
    if not is_window_active("Notepad"):
        active = get_active_window_title()
        raise RuntimeError(f"Notepad not active before close. Current: {active!r}")
    press_hotkey("ctrl", "shift", "w")
    logger.info("[%s] Close triggered", filename)
    sleep(1.0)


def close_wrong_window(*, filename: str) -> None:
    logger.info("[%s] Closing window with Alt+F4 + Escape", filename)
    close_window_hard()
    sleep(2.0)
