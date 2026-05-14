from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from notepad_grounding import desktop_interactions as steps
from notepad_grounding.api import fetch_posts
from notepad_grounding.capture import capture_desktop
from notepad_grounding.desktop import clear_target_directory
from notepad_grounding.desktop import ensure_directory
from notepad_grounding.desktop import get_target_directory
from notepad_grounding.locate import VisualSearchResult
from notepad_grounding.locate import run_locate
from notepad_grounding.models import DesktopReviewResult
from notepad_grounding.reviewers import OpenAIReviewer
from notepad_grounding.vision import OpenAIVisionClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlowDependencies:
    client: OpenAIVisionClient
    reviewer: OpenAIReviewer


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
        return _run_locate_flow(
            query=query,
            dependencies=dependencies,
            output_root=output_root,
            timestamp=timestamp,
            llm_rounds=llm_rounds,
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
    return FlowDependencies(
        client=OpenAIVisionClient(),
        reviewer=OpenAIReviewer(),
    )


def _run_locate_flow(
    *,
    query: str,
    dependencies: FlowDependencies,
    output_root: Path,
    timestamp: str | None,
    llm_rounds: int,
) -> VisualSearchResult:
    image = capture_desktop()
    return run_locate(
        image,
        query=query,
        client=dependencies.client,
        target_reviewer=dependencies.reviewer,
        bbox_reviewer=dependencies.reviewer,
        output_root=output_root / "locate",
        timestamp=timestamp,
        rounds=llm_rounds,
    )


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

    logger.info("Starting automation for query=%r", query)
    logger.info("Fetching %d posts from JSONPlaceholder...", post_limit)
    posts = fetch_posts(limit=post_limit)
    logger.info("Fetched %d posts", len(posts))

    target_dir = get_target_directory()
    ensure_directory(target_dir)
    clear_target_directory()
    logger.info("Cleared target directory: %s", target_dir)

    post_results: list[PostResult] = []
    succeeded = 0
    failed = 0

    for post in posts:
        post_result, did_succeed = _process_post(
            post=post,
            query=query,
            dependencies=dependencies,
            output_dir=output_dir,
            target_dir=target_dir,
            max_retries=max_retries,
            retry_delay=retry_delay,
            llm_rounds=llm_rounds,
        )
        post_results.append(post_result)
        if did_succeed:
            succeeded += 1
        else:
            failed += 1

    result_path = output_dir / "result.json"
    payload = {
        "query": query,
        "total_posts": len(posts),
        "succeeded": succeeded,
        "failed": failed,
        "output_dir": str(output_dir),
        "results": [asdict(result) for result in post_results],
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return AutomationResult(
        query=query,
        total_posts=len(posts),
        succeeded=succeeded,
        failed=failed,
        output_dir=str(output_dir),
        result_json=str(result_path),
    )


def _process_post(
    *,
    post: dict,
    query: str,
    dependencies: FlowDependencies,
    output_dir: Path,
    target_dir: Path,
    max_retries: int,
    retry_delay: float,
    llm_rounds: int,
) -> tuple[PostResult, bool]:
    post_id = post["id"]
    title = post["title"]
    body = post["body"]
    filename = f"post_{post_id}.txt"
    full_path = target_dir / filename
    center: tuple[int, int] | None = None
    error_msg: str | None = None
    status = "failed"

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("[%s] Attempt %d/%d: grounding icon...", filename, attempt, max_retries)
            center = _locate_and_open_target(
                query=query,
                filename=filename,
                dependencies=dependencies,
                output_dir=output_dir,
                llm_rounds=llm_rounds,
            )
            _type_review_save_and_close(
                filename=filename,
                title=title,
                body=body,
                full_path=full_path,
                dependencies=dependencies,
            )
            status = "success"
            break
        except Exception as exc:
            error_msg = str(exc)
            logger.warning("[%s] Attempt %d failed: %s", filename, attempt, error_msg)
            if attempt < max_retries:
                logger.info("[%s] Retrying in %.1fs...", filename, retry_delay)
                steps.sleep(retry_delay)
            else:
                logger.error("[%s] All %d attempts failed", filename, max_retries)

    return (
        PostResult(
            post_id=post_id,
            status=status,
            filename=filename,
            center=center,
            error=error_msg,
        ),
        status == "success",
    )


def _locate_and_open_target(
    *,
    query: str,
    filename: str,
    dependencies: FlowDependencies,
    output_dir: Path,
    llm_rounds: int,
) -> tuple[int, int]:
    ground_start = time.perf_counter()
    image = capture_desktop()
    result = run_locate(
        image,
        query=query,
        client=dependencies.client,
        target_reviewer=dependencies.reviewer,
        bbox_reviewer=dependencies.reviewer,
        output_root=output_dir / "locate",
        rounds=llm_rounds,
    )
    center = result.center
    logger.info("[%s] Icon found at %s (took %.2fs)", filename, center, result.elapsed_seconds)

    steps.open_target_icon(filename=filename, query=query, center=center)
    logger.info("[%s] Total grounding+click: %.2fs", filename, time.perf_counter() - ground_start)

    review = _review_current_desktop(
        dependencies.reviewer,
        action=f"Double-clicked the '{query}' desktop icon at {center}",
        expected="Notepad window is open and active",
    )
    if review.status == "wrong_app":
        logger.warning("[%s] Reviewer detected wrong app: %s", filename, review.rationale)
        steps.close_wrong_window(filename=filename)
        raise RuntimeError(f"Wrong app opened: {review.rationale}")
    if review.status in ("error", "retry"):
        steps.handle_recovery(review.action_needed)
        raise RuntimeError(f"Open failed: {review.rationale}")
    logger.info("[%s] Reviewer confirmed: Notepad is open", filename)
    return center


def _type_review_save_and_close(
    *,
    filename: str,
    title: str,
    body: str,
    full_path: Path,
    dependencies: FlowDependencies,
) -> None:
    steps.type_post_content(filename=filename, title=title, body=body)
    review = _review_current_desktop(
        dependencies.reviewer,
        action="Typed post content into Notepad",
        expected="Notepad shows the typed post text",
    )
    if review.status != "success":
        logger.warning("[%s] Reviewer detected typing issue: %s", filename, review.rationale)
    else:
        logger.info("[%s] Reviewer confirmed: text is correct", filename)

    steps.save_post_file(filename=filename, full_path=full_path)
    _review_save(filename=filename, full_path=full_path, desktop_reviewer=dependencies.reviewer)

    steps.close_notepad(filename=filename)
    review = _review_current_desktop(
        dependencies.reviewer,
        action="Closed Notepad",
        expected="Notepad is closed, desktop is visible",
    )
    if review.status != "success":
        logger.warning("[%s] Reviewer detected close issue: %s", filename, review.rationale)
        steps.handle_recovery(review.action_needed)
    else:
        logger.info("[%s] Reviewer confirmed: Notepad closed", filename)


def _review_save(*, filename: str, full_path: Path, desktop_reviewer) -> None:
    for _ in range(3):
        review = _review_current_desktop(
            desktop_reviewer,
            action=f"Pressed Save with path {full_path}",
            expected="File is saved, no dialogs remain, Notepad is active",
        )
        if review.status == "success":
            logger.info("[%s] Reviewer confirmed: save succeeded", filename)
            return
        if review.status == "pop_up":
            logger.info("[%s] Reviewer detected pop-up: %s", filename, review.rationale)
            steps.handle_recovery(review.action_needed)
            steps.sleep(1.0)
            continue
        if review.status == "wrong_app":
            logger.warning("[%s] Reviewer detected wrong window: %s", filename, review.rationale)
            steps.handle_recovery(review.action_needed)
            raise RuntimeError(f"Save went wrong: {review.rationale}")
        steps.handle_recovery(review.action_needed)
        return
    raise RuntimeError("Pop-up handling exceeded max cycles")


def _review_current_desktop(
    desktop_reviewer,
    *,
    action: str,
    expected: str,
) -> DesktopReviewResult:
    return steps.review_desktop_state(
        desktop_reviewer,
        action=action,
        expected=expected,
        image=capture_desktop(),
    )
