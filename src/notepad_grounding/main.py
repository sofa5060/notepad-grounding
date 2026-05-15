from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from notepad_grounding.api import fetch_posts
from notepad_grounding.automation import automate_post
from notepad_grounding.automation import reset_target_directory
from notepad_grounding.llm import locate_icon

QUERY = "Notepad"
RUNS = 10
DELAY_BETWEEN_RUNS_SECONDS = 5
OUTPUT_DIR = Path("output/notepad_grounding")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    load_env_file()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Add it to .env or set it in the shell.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = Path.home() / "Desktop" / "tjm-project"
    logger.info("clearing notes folder: %s", target_dir)
    reset_target_directory(target_dir)

    posts = fetch_posts(limit=RUNS)
    for index, post in enumerate(posts, start=1):
        run_output_dir = OUTPUT_DIR / f"{index:02d}"
        logger.info("[%d/%d] locating %s", index, len(posts), QUERY)

        guess = locate_icon(query=QUERY, output_dir=run_output_dir)
        logger.info("double-clicking %s at (%d, %d)", QUERY, guess.x, guess.y)
        full_path = automate_post(post=post, index=index, target_dir=target_dir, click_x=guess.x, click_y=guess.y)
        logger.info("saved to %s", full_path)
        if index < len(posts):
            logger.info("waiting %d seconds before next run", DELAY_BETWEEN_RUNS_SECONDS)
            time.sleep(DELAY_BETWEEN_RUNS_SECONDS)


def load_env_file(path: Path | None = None) -> None:
    env_path = path or Path.cwd() / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = strip_quotes(value.strip())
        if key and key not in os.environ:
            os.environ[key] = value


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


if __name__ == "__main__":
    main()
