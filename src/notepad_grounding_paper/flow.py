"""Top-level orchestration: wires the LLM client/reviewer into the two modes.

- "locate": just find an icon on the current desktop (debugging aid).
- "run": full automation — fetch posts from the API, then for each post
  locate the Notepad icon, open it, type the post, and save it to a file.

The GUI interactions (clicking, typing, screenshots) live in gui.py; the
icon search lives in locate.py; the LLM wiring lives in llm.py.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from notepad_grounding_paper import gui
from notepad_grounding_paper.api import fetch_posts
from notepad_grounding_paper.locate import run_locate
from notepad_grounding_paper.llm import OpenAIReviewer
from notepad_grounding_paper.llm import OpenAIVisionClient
from notepad_grounding_paper.models import AutomationResult
from notepad_grounding_paper.models import FlowDependencies
from notepad_grounding_paper.models import VisualSearchResult

logger = logging.getLogger(__name__)


def run_flow(
    *,
    mode: str,
    query: str,
    output_root: Path,
    timestamp: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    post_limit: int = 10,
    llm_rounds: int = 3,
) -> AutomationResult | VisualSearchResult:
    dependencies = build_default_dependencies()
    if mode == "locate":
        return run_locate(
            gui.capture_desktop(),
            query=query,
            client=dependencies.client,
            target_reviewer=dependencies.reviewer,
            output_root=output_root / "locate",
            timestamp=timestamp,
            rounds=llm_rounds,
        )
    if mode == "run":
        return _run_automation_flow(
            query=query,
            dependencies=dependencies,
            output_root=output_root,
            timestamp=timestamp,
            max_retries=max_retries,
            retry_delay=retry_delay,
            post_limit=post_limit,
            llm_rounds=llm_rounds,
        )
    raise ValueError(f"Unknown flow mode: {mode}")


def build_default_dependencies() -> FlowDependencies:
    return FlowDependencies(client=OpenAIVisionClient(), reviewer=OpenAIReviewer())


def _run_automation_flow(
    *,
    query: str,
    dependencies: FlowDependencies,
    output_root: Path,
    timestamp: str | None,
    max_retries: int,
    retry_delay: float,
    post_limit: int,
    llm_rounds: int,
) -> AutomationResult:
    run_id = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / "automation" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather the work: the posts to write and the folder to save them into.
    logger.info("Starting automation for query=%r", query)
    posts = fetch_posts(limit=post_limit)
    logger.info("Fetched %d posts", len(posts))

    target_dir = gui.prepare_target_directory()

    succeeded = 0
    failed = 0

    # Each post runs independently; one failure doesn't stop the rest.
    for post in posts:
        if _write_post_to_notepad(
            post=post,
            query=query,
            dependencies=dependencies,
            output_dir=output_dir,
            target_dir=target_dir,
            max_retries=max_retries,
            retry_delay=retry_delay,
            llm_rounds=llm_rounds,
        ):
            succeeded += 1
        else:
            failed += 1

    logger.info("Automation finished: %d succeeded, %d failed", succeeded, failed)
    return AutomationResult(
        query=query, total_posts=len(posts), succeeded=succeeded, failed=failed, output_dir=str(output_dir)
    )


def _write_post_to_notepad(
    *,
    post: dict,
    query: str,
    dependencies: FlowDependencies,
    output_dir: Path,
    target_dir: Path,
    max_retries: int,
    retry_delay: float,
    llm_rounds: int,
) -> bool:
    """Write one post into Notepad, retrying the whole sequence on failure.

    One attempt = locate the icon, open Notepad, type the post, save + close.
    Any exception counts as a failed attempt; after max_retries the post is
    reported as failed instead of aborting the whole run.
    """
    filename = f"post_{post['id']}.txt"
    full_path = target_dir / filename

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("[%s] Attempt %d/%d: grounding icon...", filename, attempt, max_retries)
            result = run_locate(
                gui.capture_desktop(),
                query=query,
                client=dependencies.client,
                target_reviewer=dependencies.reviewer,
                output_root=output_dir / "locate",
                rounds=llm_rounds,
            )
            logger.info("[%s] Icon found at %s (took %.2fs)", filename, result.center, result.elapsed_seconds)
            _open_target(filename=filename, query=query, center=result.center, dependencies=dependencies)
            _type_post(filename=filename, title=post["title"], body=post["body"], dependencies=dependencies)
            _save_and_close(filename=filename, full_path=full_path, dependencies=dependencies)
            return True
        except Exception as exc:
            logger.warning("[%s] Attempt %d failed: %s", filename, attempt, exc)
            if attempt < max_retries:
                gui.sleep(retry_delay)

    logger.error("[%s] All %d attempts failed", filename, max_retries)
    return False


def _open_target(*, filename: str, query: str, center: tuple[int, int], dependencies: FlowDependencies) -> None:
    """Double-click the located icon, then verify Notepad actually opened."""
    gui.double_click(*center)
    gui.sleep(2.0)
    review = dependencies.reviewer.review_desktop_state(
        action=f"Double-clicked the '{query}' desktop icon at {center}",
        expected="Notepad window is open",
        image=gui.capture_desktop(),
    )
    if review.status == "wrong_app":
        logger.warning("[%s] Reviewer detected wrong app: %s", filename, review.rationale)
        gui.close_window_hard()
        gui.sleep(2.0)
        raise RuntimeError(f"Wrong app opened: {review.rationale}")
    if review.status in ("error", "retry"):
        gui.handle_recovery(review.action_needed)
        raise RuntimeError(f"Open failed: {review.rationale}")


def _type_post(*, filename: str, title: str, body: str, dependencies: FlowDependencies) -> None:
    """Type the post into Notepad and sanity-check the screen afterwards."""
    gui.type_text(f"Title: {title}\n\n{body}")
    gui.sleep(0.5)
    review = dependencies.reviewer.review_desktop_state(
        action="Typed post content into Notepad",
        expected="Notepad shows the typed post text",
        image=gui.capture_desktop(),
    )
    if review.status != "success":
        logger.warning("[%s] Reviewer detected typing issue: %s", filename, review.rationale)


def _save_and_close(*, filename: str, full_path: Path, dependencies: FlowDependencies) -> None:
    """Save the file, clear any pop-up dialogs, then close Notepad."""
    gui.save_post_file(full_path=full_path)
    # Up to 3 review cycles to clear pop-ups (e.g. an overwrite confirmation)
    # before giving up on the save.
    for _ in range(3):
        review = dependencies.reviewer.review_desktop_state(
            action=f"Pressed Save with path {full_path}",
            expected="File is saved, no dialogs remain, Notepad is open",
            image=gui.capture_desktop(),
        )
        if review.status == "success":
            break
        if review.status == "pop_up":
            logger.info("[%s] Reviewer detected pop-up: %s", filename, review.rationale)
            gui.handle_recovery(review.action_needed)
            gui.sleep(1.0)
            continue
        if review.status == "wrong_app":
            logger.warning("[%s] Reviewer detected wrong window: %s", filename, review.rationale)
            gui.handle_recovery(review.action_needed)
            raise RuntimeError(f"Save went wrong: {review.rationale}")
        gui.handle_recovery(review.action_needed)
        break
    else:
        raise RuntimeError("Pop-up handling exceeded max cycles")

    gui.close_notepad()
    review = dependencies.reviewer.review_desktop_state(
        action="Closed Notepad", expected="Notepad is closed, desktop is visible", image=gui.capture_desktop()
    )
    if review.status != "success":
        logger.warning("[%s] Reviewer detected close issue: %s", filename, review.rationale)
        gui.handle_recovery(review.action_needed)
