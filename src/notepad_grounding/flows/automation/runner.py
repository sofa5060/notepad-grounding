from __future__ import annotations

import logging
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from notepad_grounding.flows.llm_visual_search.flow import run_llm_visual_search
from notepad_grounding.shared.api import ApiError
from notepad_grounding.shared.api import fetch_posts
from notepad_grounding.shared.automation import clear_target_directory
from notepad_grounding.shared.automation import click_at
from notepad_grounding.shared.automation import close_active_window
from notepad_grounding.shared.automation import close_window_hard
from notepad_grounding.shared.automation import double_click
from notepad_grounding.shared.automation import ensure_directory
from notepad_grounding.shared.automation import get_active_window_title
from notepad_grounding.shared.automation import get_target_directory
from notepad_grounding.shared.automation import is_window_active
from notepad_grounding.shared.automation import press_hotkey
from notepad_grounding.shared.automation import sleep
from notepad_grounding.shared.automation import type_text
from notepad_grounding.shared.automation import wait_for_window
from notepad_grounding.shared.automation import wait_for_window_close
from notepad_grounding.shared.capture import capture_desktop
from notepad_grounding.shared.llm import OpenAIVisionClient
from notepad_grounding.shared.llm import VisionClient
from notepad_grounding.shared.reviewer import OpenAIReviewClient
from notepad_grounding.shared.reviewer import ReviewClient
from notepad_grounding.shared.reviewer import ReviewResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutomationResult:
    query: str
    total_posts: int
    succeeded: int
    failed: int
    output_dir: str
    result_json: str


@dataclass(frozen=True)
class PostResult:
    post_id: int
    status: str
    filename: str
    center: tuple[int, int] | None
    error: str | None


def _review_and_recover(
    reviewer: ReviewClient,
    action: str,
    expected: str,
    image: Image.Image,
) -> ReviewResult:
    """Ask the reviewer to check the screen state and return its verdict."""
    logger.info("[REVIEW] %s | Expected: %s", action, expected)
    result = reviewer.review_state(action=action, expected=expected, image=image)
    logger.info("[REVIEW] status=%s action_needed=%s rationale=%s", result.status, result.action_needed, result.rationale)
    return result


def _handle_recovery(action_needed: str) -> None:
    """Execute a recovery action suggested by the reviewer."""
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


def run_automation(
    *,
    query: str,
    client: VisionClient,
    output_root: Path,
    timestamp: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    post_limit: int = 10,
    llm_rounds: int = 3,
    reviewer: ReviewClient | None = None,
) -> AutomationResult:
    run_id = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / "automation" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use default reviewer if none provided
    reviewer = reviewer or OpenAIReviewClient()

    logger.info("Starting automation for query=%r", query)

    # Fetch posts
    logger.info("Fetching %d posts from JSONPlaceholder...", post_limit)
    posts = fetch_posts(limit=post_limit)
    logger.info("Fetched %d posts", len(posts))

    ensure_directory(get_target_directory())
    clear_target_directory()
    logger.info("Cleared target directory: %s", get_target_directory())

    post_results: list[PostResult] = []
    succeeded = 0
    failed = 0

    for post in posts:
        post_id = post["id"]
        title = post["title"]
        body = post["body"]
        filename = f"post_{post_id}.txt"
        full_path = get_target_directory() / filename

        center: tuple[int, int] | None = None
        error_msg: str | None = None
        status = "failed"

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "[%s] Attempt %d/%d: grounding icon...",
                    filename,
                    attempt,
                    max_retries,
                )

                # === STEP 1-2: Ground icon + click (timed) ===
                ground_start = time.perf_counter()
                image = capture_desktop()
                result = run_llm_visual_search(
                    image,
                    query=query,
                    client=client,
                    output_root=output_dir / "locate",
                    rounds=llm_rounds,
                )
                center = result.center
                logger.info("[%s] Icon found at %s (took %.2fs)", filename, center, result.elapsed_seconds)

                # === STEP 2: Click ===
                double_click(*center)
                ground_elapsed = time.perf_counter() - ground_start
                logger.info("[%s] Double-clicked at %s | Total grounding+click: %.2fs", filename, center, ground_elapsed)
                sleep(2.0)

                # === STEP 3: Review — did Notepad open? ===
                review = _review_and_recover(
                    reviewer,
                    action=f"Double-clicked the '{query}' desktop icon at {center}",
                    expected="Notepad window is open and active",
                    image=capture_desktop(),
                )
                if review.status == "wrong_app":
                    logger.warning("[%s] Reviewer detected wrong app: %s", filename, review.rationale)
                    logger.info("[%s] Closing window with Alt+F4 + Escape (safe for both app and desktop)...", filename)
                    close_window_hard()
                    sleep(2.0)
                    raise RuntimeError(f"Wrong app opened: {review.rationale}")
                elif review.status in ("error", "retry"):
                    _handle_recovery(review.action_needed)
                    raise RuntimeError(f"Open failed: {review.rationale}")
                logger.info("[%s] Reviewer confirmed: Notepad is open", filename)

                # === STEP 4: Type content ===
                content = f"Title: {title}\n\n{body}"
                type_text(content)
                logger.info("[%s] Typed post content (%d chars)", filename, len(content))
                sleep(0.5)

                # === STEP 5: Review — is text correct? ===
                review = _review_and_recover(
                    reviewer,
                    action=f"Typed post content into Notepad",
                    expected="Notepad shows the typed post text",
                    image=capture_desktop(),
                )
                if review.status != "success":
                    logger.warning("[%s] Reviewer detected typing issue: %s", filename, review.rationale)
                else:
                    logger.info("[%s] Reviewer confirmed: text is correct", filename)

                # === STEP 6: Save ===
                press_hotkey("ctrl", "shift", "s")
                sleep(1.5)
                type_text(str(full_path), interval=0.01)
                sleep(0.5)
                press_hotkey("return")
                sleep(1.0)
                logger.info("[%s] Save triggered", filename)

                # === STEP 7: Review — did save succeed? Handle pop-ups ===
                for _ in range(3):  # up to 3 review cycles for pop-ups
                    review = _review_and_recover(
                        reviewer,
                        action=f"Pressed Save with path {full_path}",
                        expected="File is saved, no dialogs remain, Notepad is active",
                        image=capture_desktop(),
                    )
                    if review.status == "success":
                        logger.info("[%s] Reviewer confirmed: save succeeded", filename)
                        break
                    elif review.status == "pop_up":
                        logger.info("[%s] Reviewer detected pop-up: %s", filename, review.rationale)
                        _handle_recovery(review.action_needed)
                        sleep(1.0)
                    elif review.status == "wrong_app":
                        logger.warning("[%s] Reviewer detected wrong window: %s", filename, review.rationale)
                        _handle_recovery(review.action_needed)
                        raise RuntimeError(f"Save went wrong: {review.rationale}")
                    else:
                        _handle_recovery(review.action_needed)
                        break
                else:
                    raise RuntimeError("Pop-up handling exceeded max cycles")

                # === STEP 8: Close ===
                if not is_window_active("Notepad"):
                    active = get_active_window_title()
                    raise RuntimeError(f"Notepad not active before close. Current: {active!r}")
                press_hotkey("ctrl", "shift", "w")
                logger.info("[%s] Close triggered", filename)
                sleep(1.0)

                # === STEP 9: Review — did Notepad close? ===
                review = _review_and_recover(
                    reviewer,
                    action="Closed Notepad",
                    expected="Notepad is closed, desktop is visible",
                    image=capture_desktop(),
                )
                if review.status != "success":
                    logger.warning("[%s] Reviewer detected close issue: %s", filename, review.rationale)
                    _handle_recovery(review.action_needed)
                else:
                    logger.info("[%s] Reviewer confirmed: Notepad closed", filename)

                status = "success"
                succeeded += 1
                break

            except Exception as exc:
                error_msg = str(exc)
                logger.warning(
                    "[%s] Attempt %d failed: %s",
                    filename,
                    attempt,
                    error_msg,
                )
                if attempt < max_retries:
                    logger.info("[%s] Retrying in %.1fs...", filename, retry_delay)
                    sleep(retry_delay)
                else:
                    logger.error("[%s] All %d attempts failed", filename, max_retries)
                    failed += 1

        post_results.append(
            PostResult(
                post_id=post_id,
                status=status,
                filename=filename,
                center=center,
                error=error_msg,
            )
        )

    # Save results
    result_path = output_dir / "result.json"
    payload = {
        "query": query,
        "total_posts": len(posts),
        "succeeded": succeeded,
        "failed": failed,
        "output_dir": str(output_dir),
        "results": [asdict(r) for r in post_results],
    }
    result_path.write_text(
        __import__("json").dumps(payload, indent=2),
        encoding="utf-8",
    )

    return AutomationResult(
        query=query,
        total_posts=len(posts),
        succeeded=succeeded,
        failed=failed,
        output_dir=str(output_dir),
        result_json=str(result_path),
    )
