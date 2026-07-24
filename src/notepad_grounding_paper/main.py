from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from notepad_grounding_paper.api import fetch_posts
from notepad_grounding_paper.automation import automate_post
from notepad_grounding_paper.llm import capture_desktop
from notepad_grounding_paper.locate import run_locate

QUERY = "Notepad"
POST_LIMIT = 10
POST_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0
OUTPUT_ROOT = Path("output/notepad_grounding_paper")
TARGET_DIR = Path.home() / "Desktop" / "tjm-project"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def reset_target_directory(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in target_dir.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def main() -> None:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Add it to .env or set it in the shell.")

    if "locate" in sys.argv[1:]:
        center = run_locate(capture_desktop(), query=QUERY, output_root=OUTPUT_ROOT / "locate")
        print(f"center={center[0]},{center[1]}")
        return

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = OUTPUT_ROOT / "automation" / run_id
    reset_target_directory(TARGET_DIR)
    posts = fetch_posts(limit=POST_LIMIT)

    succeeded = 0
    for post in posts:
        for attempt in range(1, POST_ATTEMPTS + 1):
            try:
                center = run_locate(capture_desktop(), query=QUERY, output_root=output_dir / "locate")
                full_path = automate_post(post=post, query=QUERY, center=center, target_dir=TARGET_DIR)
                logger.info("[post %s] saved to %s", post["id"], full_path)
                succeeded += 1
                break
            except Exception as exc:
                logger.warning("[post %s] attempt %d/%d failed: %s", post["id"], attempt, POST_ATTEMPTS, exc)
                if attempt < POST_ATTEMPTS:
                    time.sleep(RETRY_DELAY_SECONDS)
        else:
            logger.error("[post %s] all %d attempts failed", post["id"], POST_ATTEMPTS)

    logger.info("done: %d/%d posts succeeded", succeeded, len(posts))


if __name__ == "__main__":
    main()
