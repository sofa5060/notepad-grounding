from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from notepad_grounding_paper import desktop as steps
from notepad_grounding_paper.api import fetch_posts
from notepad_grounding_paper.locate import run_locate
from notepad_grounding_paper.llm import OpenAIReviewer
from notepad_grounding_paper.llm import OpenAIVisionClient
from notepad_grounding_paper.models import AutomationResult
from notepad_grounding_paper.models import FlowDependencies
from notepad_grounding_paper.models import PostResult
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
            steps.capture_desktop(),
            query=query,
            client=dependencies.client,
            target_reviewer=dependencies.reviewer,
            bbox_reviewer=dependencies.reviewer,
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
    return FlowDependencies(
        client=OpenAIVisionClient(),
        reviewer=OpenAIReviewer(),
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
    posts = fetch_posts(limit=post_limit)
    logger.info("Fetched %d posts", len(posts))

    target_dir = steps.get_target_directory()
    steps.ensure_directory(target_dir)
    steps.clear_target_directory()

    post_results: list[PostResult] = []
    succeeded = 0
    failed = 0

    for post in posts:
        post_result, did_succeed = _write_post_to_notepad(
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
            result = run_locate(
                steps.capture_desktop(),
                query=query,
                client=dependencies.client,
                target_reviewer=dependencies.reviewer,
                bbox_reviewer=dependencies.reviewer,
                output_root=output_dir / "locate",
                rounds=llm_rounds,
            )
            center = result.center
            logger.info("[%s] Icon found at %s (took %.2fs)", filename, center, result.elapsed_seconds)
            _open_target(filename=filename, query=query, center=center, dependencies=dependencies)
            _type_post(filename=filename, title=title, body=body, dependencies=dependencies)
            _save_and_close(filename=filename, full_path=full_path, dependencies=dependencies)
            status = "success"
            break
        except Exception as exc:
            error_msg = str(exc)
            logger.warning("[%s] Attempt %d failed: %s", filename, attempt, error_msg)
            if attempt < max_retries:
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


def _open_target(
    *,
    filename: str,
    query: str,
    center: tuple[int, int],
    dependencies: FlowDependencies,
) -> None:
    steps.double_click(*center)
    steps.sleep(2.0)
    review = dependencies.reviewer.review_desktop_state(
        action=f"Double-clicked the '{query}' desktop icon at {center}",
        expected="Notepad window is open",
        image=steps.capture_desktop(),
    )
    if review.status == "wrong_app":
        logger.warning("[%s] Reviewer detected wrong app: %s", filename, review.rationale)
        steps.close_window_hard()
        steps.sleep(2.0)
        raise RuntimeError(f"Wrong app opened: {review.rationale}")
    if review.status in ("error", "retry"):
        steps.handle_recovery(review.action_needed)
        raise RuntimeError(f"Open failed: {review.rationale}")


def _type_post(
    *,
    filename: str,
    title: str,
    body: str,
    dependencies: FlowDependencies,
) -> None:
    steps.type_text(f"Title: {title}\n\n{body}")
    steps.sleep(0.5)
    review = dependencies.reviewer.review_desktop_state(
        action="Typed post content into Notepad",
        expected="Notepad shows the typed post text",
        image=steps.capture_desktop(),
    )
    if review.status != "success":
        logger.warning("[%s] Reviewer detected typing issue: %s", filename, review.rationale)


def _save_and_close(
    *,
    filename: str,
    full_path: Path,
    dependencies: FlowDependencies,
) -> None:
    steps.save_post_file(full_path=full_path)
    for _ in range(3):
        review = dependencies.reviewer.review_desktop_state(
            action=f"Pressed Save with path {full_path}",
            expected="File is saved, no dialogs remain, Notepad is open",
            image=steps.capture_desktop(),
        )
        if review.status == "success":
            break
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
        break
    else:
        raise RuntimeError("Pop-up handling exceeded max cycles")

    steps.close_notepad()
    review = dependencies.reviewer.review_desktop_state(
        action="Closed Notepad",
        expected="Notepad is closed, desktop is visible",
        image=steps.capture_desktop(),
    )
    if review.status != "success":
        logger.warning("[%s] Reviewer detected close issue: %s", filename, review.rationale)
        steps.handle_recovery(review.action_needed)


