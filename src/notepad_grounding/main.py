from __future__ import annotations
import logging
import os
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
from notepad_grounding.api import fetch_posts
from notepad_grounding.automation import automate_post
from notepad_grounding.llm import locate_icon

QUERY = (
    "Windows Notepad desktop shortcut icon. Match the Notepad icon by visual "
    "appearance even if the visible desktop label is different."
)
RUNS = 10
DELAY_BETWEEN_RUNS_SECONDS = 5
OUTPUT_DIR = Path("output/notepad_grounding")

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

    # Ensure output and target directories exist and are clean
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = Path.home() / "Desktop" / "tjm-project"
    logger.info("clearing notes folder: %s", target_dir)
    reset_target_directory(target_dir)

    posts = fetch_posts(limit=RUNS)
    for index, post in enumerate(posts, start=1):
        run_output_dir = OUTPUT_DIR / f"{index:02d}"
        logger.info("[%d/%d] locating %s", index, len(posts), QUERY)

        coordinates = locate_icon(query=QUERY, output_dir=run_output_dir)
        if coordinates is None:
            logger.warning("[%d/%d] could not locate %s with sufficient confidence, skipping",
                           index, len(posts), QUERY)
            continue
        logger.info("double-clicking %s at (%d, %d)", QUERY, coordinates.x, coordinates.y)
        full_path = automate_post(post=post, index=index, target_dir=target_dir, click_x=coordinates.x, click_y=coordinates.y)
        logger.info("saved to %s", full_path)
        
        # Wait a bit before the next run to allow for moving the icon if needed
        if index < len(posts):
            logger.info("waiting %d seconds before next run", DELAY_BETWEEN_RUNS_SECONDS)
            time.sleep(DELAY_BETWEEN_RUNS_SECONDS)


if __name__ == "__main__":
    main()
