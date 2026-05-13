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
from notepad_grounding.shared.automation import double_click
from notepad_grounding.shared.automation import ensure_directory
from notepad_grounding.shared.automation import file_exists
from notepad_grounding.shared.automation import get_target_directory
from notepad_grounding.shared.automation import press_hotkey
from notepad_grounding.shared.automation import sleep
from notepad_grounding.shared.automation import type_text
from notepad_grounding.shared.capture import capture_desktop
from notepad_grounding.shared.llm import OpenAIVisionClient
from notepad_grounding.shared.llm import VisionClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutomationResult:
    query: str
    total_posts: int
    succeeded: int
    failed: int
    skipped: int
    output_dir: str
    result_json: str


@dataclass(frozen=True)
class PostResult:
    post_id: int
    status: str  # "success", "failed", "skipped"
    filename: str
    center: tuple[int, int] | None
    error: str | None


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
) -> AutomationResult:
    run_id = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / "automation" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch posts
    logger.info("Fetching %d posts from JSONPlaceholder...", post_limit)
    posts = fetch_posts(limit=post_limit)
    logger.info("Fetched %d posts", len(posts))

    # Ensure target directory exists
    ensure_directory(get_target_directory())

    post_results: list[PostResult] = []
    succeeded = 0
    failed = 0
    skipped = 0

    for post in posts:
        post_id = post["id"]
        title = post["title"]
        body = post["body"]
        filename = f"post_{post_id}.txt"
        full_path = get_target_directory() / filename

        # Skip if already exists
        if file_exists(filename):
            logger.info("[%s] Already exists, skipping", filename)
            post_results.append(
                PostResult(
                    post_id=post_id,
                    status="skipped",
                    filename=filename,
                    center=None,
                    error=None,
                )
            )
            skipped += 1
            continue

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

                # 1. Capture screenshot and locate icon
                image = capture_desktop()
                result = run_llm_visual_search(
                    image,
                    query=query,
                    client=client,
                    output_root=output_dir / "locate",
                    rounds=llm_rounds,
                )
                center = result.center
                logger.info("[%s] Icon found at %s", filename, center)

                # 2. Double-click to launch Notepad
                double_click(*center)
                logger.info("[%s] Double-clicked at %s", filename, center)

                # 3. Wait for Notepad to open
                sleep(2.0)

                # 4. Type content
                content = f"Title: {title}\n\n{body}"
                type_text(content)
                logger.info("[%s] Typed post content", filename)

                # 5. Save file
                press_hotkey("ctrl", "s")
                sleep(0.5)
                # Type full absolute path
                type_text(str(full_path))
                sleep(0.2)
                press_hotkey("return")
                logger.info("[%s] Saved to %s", filename, full_path)

                # 6. Wait for save then close Notepad
                sleep(1.0)
                press_hotkey("alt", "f4")
                logger.info("[%s] Closed Notepad", filename)

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
        "skipped": skipped,
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
        skipped=skipped,
        output_dir=str(output_dir),
        result_json=str(result_path),
    )
