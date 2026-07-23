from __future__ import annotations

import logging
from pathlib import Path

import mss
from PIL import Image

from grid_ocr.annotate import draw_candidates
from grid_ocr.annotate import draw_ocr
from grid_ocr.grounding import find_candidates
from grid_ocr.grounding import locate_from_lines
from grid_ocr.ocr import extract_ocr_lines_from_grid

QUERY = "Notepad"
OUT_DIR = Path("output/grid_ocr")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def capture_desktop() -> Image.Image:
    with mss.MSS() as screen_capture:
        screenshot = screen_capture.grab(screen_capture.monitors[1])
    return Image.frombytes("RGB", screenshot.size, screenshot.rgb)


def main() -> None:
    # 1. Capture the desktop.
    image = capture_desktop()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUT_DIR / "screenshot.png")

    # 2. Run Windows OCR over the overlapping tile grid.
    logger.info("running Windows OCR over the tile grid")
    lines = extract_ocr_lines_from_grid(image)
    draw_ocr(image, lines, OUT_DIR / "ocr.png")

    # 3. Score OCR lines against the query and pick the best icon candidate.
    result = locate_from_lines(lines, screen_size=image.size, query=QUERY)
    draw_candidates(
        image,
        candidates=find_candidates(lines, screen_size=image.size, query=QUERY),
        selected=result,
        output_path=OUT_DIR / "candidates.png",
    )

    if result is None:
        logger.warning("no candidate matched %r", QUERY)
        raise SystemExit(1)
    (x, y) = result.icon_center
    logger.info("located %r at (%d, %d)", result.label_text, x, y)
    print(f"{x},{y}")


if __name__ == "__main__":
    main()
