from __future__ import annotations
import ctypes
from ctypes import wintypes
import logging
import platform
import time
from pathlib import Path
import pyautogui
from notepad_grounding.llm import capture_desktop
from notepad_grounding.llm import verify_app_opened

WINDOW_OPEN_TIMEOUT_SECONDS = 8.0
WINDOW_CLOSE_TIMEOUT_SECONDS = 5.0
WINDOW_POLL_SECONDS = 0.25
WINDOW_SETTLE_SECONDS = 0.75
TEXT_TYPE_INTERVAL_SECONDS = 0.01
PATH_TYPE_INTERVAL_SECONDS = 0.01

logger = logging.getLogger(__name__)


class NotepadOpenTimeoutError(RuntimeError):
    pass


class AppOpenVerificationError(RuntimeError):
    pass


class AppCloseTimeoutError(RuntimeError):
    pass


def click_icon(*, x: int, y: int, expected_app: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_image = capture_desktop()
    before_image.save(output_dir / "open-before.png")
    before_windows = visible_window_handles()
    pyautogui.moveTo(x, y, duration=0.3)
    pyautogui.doubleClick()
    opened = wait_for_visible_window_change(before_windows, action="open", timeout_seconds=WINDOW_OPEN_TIMEOUT_SECONDS)
    if before_windows is not None and not opened:
        raise NotepadOpenTimeoutError(
            f"Timed out after {WINDOW_OPEN_TIMEOUT_SECONDS:.1f}s waiting for {expected_app} to open"
        )

    after_image = capture_desktop()
    after_image.save(output_dir / "open-after.png")
    review = verify_app_opened(
        expected_app=expected_app,
        before_image=before_image,
        after_image=after_image,
    )
    if review is not None:
        (output_dir / "open-verification.json").write_text(
            review.model_dump_json(indent=2),
            encoding="utf-8",
        )
    if review is not None and review.opened_expected_app:
        return

    if not close_foreground_app():
        raise AppCloseTimeoutError(
            f"Timed out after {WINDOW_CLOSE_TIMEOUT_SECONDS:.1f}s closing the unverified app"
        )
    raise AppOpenVerificationError(f"Visual verification did not confirm {expected_app}")


def close_foreground_app() -> bool:
    before_windows = visible_window_handles()
    logger.info("closing unverified foreground app with Alt+F4")
    pyautogui.hotkey("alt", "f4")
    return wait_for_visible_window_change(
        before_windows,
        action="Alt+F4 close",
        timeout_seconds=WINDOW_CLOSE_TIMEOUT_SECONDS,
    )


def type_content(*, content: str) -> None:
    logger.info("typing %d characters into Notepad", len(content))
    pyautogui.write(content, interval=TEXT_TYPE_INTERVAL_SECONDS)
    time.sleep(0.5)


def save_file(*, full_path: Path) -> None:
    full_path.parent.mkdir(parents=True, exist_ok=True)
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


def automate_post(
    *,
    post: dict,
    index: int,
    target_dir: Path,
    click_x: int,
    click_y: int,
    expected_app: str,
    output_dir: Path,
) -> Path:
    post_id = post.get("id", index)
    full_path = target_dir / f"post_{post_id}.txt"
    title = str(post.get("title", "")).strip()
    body = str(post.get("body", "")).strip()
    content = f"Title: {title}\n\n{body}"

    click_icon(x=click_x, y=click_y, expected_app=expected_app, output_dir=output_dir)
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
