from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from notepad_grounding.flows.grid_ocr.annotate import draw_candidates
from notepad_grounding.flows.grid_ocr.annotate import draw_grid
from notepad_grounding.flows.grid_ocr.annotate import draw_ocr
from notepad_grounding.flows.grid_ocr.grounding import infer_candidates
from notepad_grounding.flows.grid_ocr.grounding import locate_from_lines
from notepad_grounding.flows.grid_ocr.ocr import OcrError
from notepad_grounding.flows.grid_ocr.ocr import extract_windows_ocr_lines
from notepad_grounding.flows.grid_ocr.ocr import iter_ocr_tiles
from notepad_grounding.flows.automation.runner import run_automation
from notepad_grounding.flows.llm_visual_search.flow import run_llm_visual_search
from notepad_grounding.shared.api import ApiError
from notepad_grounding.shared.capture import CaptureError
from notepad_grounding.shared.capture import capture_desktop
from notepad_grounding.shared.capture import load_image
from notepad_grounding.shared.llm import OpenAIVisionClient


def _setup_logging() -> None:
    """Configure logging so INFO-level messages go to stderr."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "locate":
        return run_locate(args)
    if args.command == "automate":
        _setup_logging()
        return run_automate(args)
    parser.print_help()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="notepad-grounding",
        description="Query-based Windows desktop icon grounding proof.",
    )
    subparsers = parser.add_subparsers(dest="command")

    locate = subparsers.add_parser("locate", help="Locate a desktop icon by visible query.")
    locate.add_argument("--query", required=True, help="Visible target, for example Notepad.")
    locate.add_argument(
        "--flow",
        choices=("llm-visual", "grid-ocr"),
        default="llm-visual",
        help="Grounding flow to run. Default uses LLM visual search.",
    )
    locate.add_argument("--image", type=Path, default=None, help="Replay an existing screenshot instead of capturing.")
    locate.add_argument("--out-dir", type=Path, default=Path("output"), help="Debug output root.")
    locate.add_argument("--allow-non-windows", action="store_true", help="Allow live capture outside Windows.")

    locate.add_argument("--taskbar-height", type=int, default=48, help=argparse.SUPPRESS)
    locate.add_argument("--ocr-mode", choices=("grid", "full"), default="grid", help=argparse.SUPPRESS)
    locate.add_argument("--ocr-scale", type=int, default=2, help=argparse.SUPPRESS)
    locate.add_argument("--ocr-tile-width", type=int, default=320, help=argparse.SUPPRESS)
    locate.add_argument("--ocr-tile-height", type=int, default=240, help=argparse.SUPPRESS)
    locate.add_argument("--ocr-tile-overlap", type=int, default=48, help=argparse.SUPPRESS)
    locate.add_argument("--llm-rounds", type=int, default=3, help=argparse.SUPPRESS)
    locate.add_argument("--llm-model", type=str, default=None, help=argparse.SUPPRESS)

    automate = subparsers.add_parser("automate", help="Launch Notepad and save posts from JSONPlaceholder.")
    automate.add_argument("--query", required=True, help="Visible target, for example Notepad.")
    automate.add_argument("--out-dir", type=Path, default=Path("output"), help="Debug output root.")
    automate.add_argument("--max-retries", type=int, default=3, help="Retry attempts per post.")
    automate.add_argument("--retry-delay", type=float, default=1.0, help="Seconds between retries.")
    automate.add_argument("--post-limit", type=int, default=10, help="Number of posts to fetch.")
    automate.add_argument("--llm-rounds", type=int, default=3, help=argparse.SUPPRESS)
    automate.add_argument("--llm-model", type=str, default=None, help=argparse.SUPPRESS)
    return parser


def run_locate(args: argparse.Namespace) -> int:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        image, source_path = _load_or_capture(args, timestamp)
    except (CaptureError, OSError) as exc:
        print(f"error: {exc}")
        return 2

    if args.flow == "llm-visual":
        return run_llm_visual_locate(args, image)
    return run_grid_ocr_locate(args, image, source_path, timestamp)


def run_automate(args: argparse.Namespace) -> int:
    try:
        client = OpenAIVisionClient(model=args.llm_model)
        result = run_automation(
            query=args.query,
            client=client,
            output_root=args.out_dir,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            post_limit=args.post_limit,
            llm_rounds=args.llm_rounds,
        )
    except ApiError as exc:
        print(f"error: {exc}")
        return 4
    except Exception as exc:
        print(f"error: {exc}")
        return 3

    print(f"flow=automate")
    print(f"output_dir={result.output_dir}")
    print(f"result={result.result_json}")
    print(f"total_posts={result.total_posts}")
    print(f"succeeded={result.succeeded}")
    print(f"failed={result.failed}")
    print(f"skipped={result.skipped}")
    return 0


def run_llm_visual_locate(args: argparse.Namespace, image) -> int:
    try:
        client = OpenAIVisionClient(model=args.llm_model)
        result = run_llm_visual_search(
            image,
            query=args.query,
            client=client,
            output_root=args.out_dir / "llm_visual_search",
            rounds=args.llm_rounds,
        )
    except Exception as exc:
        print(f"error: {exc}")
        return 3

    print(f"flow=llm-visual")
    print(f"output_dir={result.output_dir}")
    print(f"result={result.result_json}")
    print(f"found=true")
    print(f"center={result.center[0]},{result.center[1]}")
    return 0


def run_grid_ocr_locate(args: argparse.Namespace, image, source_path: Path, timestamp: str) -> int:
    out_dir = args.out_dir / "grid_ocr"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        lines = extract_windows_ocr_lines(
            image,
            mode=args.ocr_mode,
            scale=args.ocr_scale,
            tile_width=args.ocr_tile_width,
            tile_height=args.ocr_tile_height,
            overlap=args.ocr_tile_overlap,
        )
    except OcrError as exc:
        print(f"error: {exc}")
        return 3

    grid = iter_ocr_tiles(
        width=image.width,
        height=max(0, image.height - args.taskbar_height),
        tile_width=args.ocr_tile_width,
        tile_height=args.ocr_tile_height,
        overlap=args.ocr_tile_overlap,
    )
    result = locate_from_lines(
        lines,
        screen_size=image.size,
        query=args.query,
        taskbar_height=args.taskbar_height,
    )
    candidates = result.candidates if result else infer_candidates(
        lines,
        screen_size=image.size,
        query=args.query,
        taskbar_height=args.taskbar_height,
    )
    selected = result.candidate if result else None

    grid_path = out_dir / f"{timestamp}-grid.png"
    ocr_path = out_dir / f"{timestamp}-ocr.png"
    candidates_path = out_dir / f"{timestamp}-candidates.png"
    result_path = out_dir / f"{timestamp}-result.json"

    draw_grid(image, grid, grid_path)
    draw_ocr(image, lines, ocr_path)
    draw_candidates(image, candidates=candidates, selected=selected, output_path=candidates_path)

    payload = {
        "query": args.query,
        "source": str(source_path),
        "screen_size": list(image.size),
        "flow": "grid-ocr",
        "found": result is not None,
        "center": list(result.center) if result else None,
        "selected_candidate": asdict(result.candidate) if result else None,
        "candidate_count": len(candidates),
        "outputs": {
            "grid": str(grid_path),
            "ocr": str(ocr_path),
            "candidates": str(candidates_path),
        },
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("flow=grid-ocr")
    print(f"source={source_path}")
    print(f"grid={grid_path}")
    print(f"ocr={ocr_path}")
    print(f"candidates={candidates_path}")
    print(f"result={result_path}")
    if result is None:
        print("found=false")
        return 1
    print("found=true")
    print(f"center={result.center[0]},{result.center[1]}")
    return 0


def _load_or_capture(args: argparse.Namespace, timestamp: str):
    if args.image:
        return load_image(args.image), args.image

    image = capture_desktop(require_windows=not args.allow_non_windows)
    out_dir = args.out_dir / args.flow.replace("-", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = out_dir / f"{timestamp}-raw.png"
    image.save(source_path)
    return image, source_path


if __name__ == "__main__":
    raise SystemExit(main())
