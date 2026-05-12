from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from notepad_grounding.annotate import draw_candidates
from notepad_grounding.annotate import draw_grid
from notepad_grounding.annotate import draw_ocr
from notepad_grounding.capture import CaptureError
from notepad_grounding.capture import capture_desktop
from notepad_grounding.capture import load_image
from notepad_grounding.grounding import build_grid
from notepad_grounding.grounding import infer_candidates
from notepad_grounding.grounding import locate_from_lines
from notepad_grounding.ocr import OcrError
from notepad_grounding.ocr import extract_windows_ocr_lines


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "locate":
        return run_locate(args)
    parser.print_help()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="notepad-grounding",
        description="Query-based Windows desktop icon grounding proof.",
    )
    subparsers = parser.add_subparsers(dest="command")

    locate = subparsers.add_parser("locate", help="Locate a desktop icon by visible label query.")
    locate.add_argument("--query", required=True, help="Visible target label, for example Notepad.")
    locate.add_argument("--image", type=Path, default=None, help="Replay an existing screenshot instead of capturing.")
    locate.add_argument("--out-dir", type=Path, default=Path("output/debug"), help="Debug output directory.")
    locate.add_argument("--cell-width", type=int, default=96, help="Debug grid cell width.")
    locate.add_argument("--cell-height", type=int, default=96, help="Debug grid cell height.")
    locate.add_argument("--taskbar-height", type=int, default=48, help="Bottom taskbar exclusion height.")
    locate.add_argument("--allow-non-windows", action="store_true", help="Allow live capture outside Windows.")
    return parser


def run_locate(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        if args.image:
            image = load_image(args.image)
            source_path = args.image
        else:
            image = capture_desktop(require_windows=not args.allow_non_windows)
            source_path = args.out_dir / f"{timestamp}-raw.png"
            image.save(source_path)
    except (CaptureError, OSError) as exc:
        print(f"error: {exc}")
        return 2

    try:
        lines = extract_windows_ocr_lines(image)
    except OcrError as exc:
        print(f"error: {exc}")
        return 3

    grid = build_grid(
        image.size,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        taskbar_height=args.taskbar_height,
    )
    result = locate_from_lines(
        lines,
        screen_size=image.size,
        query=args.query,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        taskbar_height=args.taskbar_height,
    )
    candidates = result.candidates if result else infer_candidates(
        lines,
        screen_size=image.size,
        query=args.query,
        taskbar_height=args.taskbar_height,
    )
    selected = result.candidate if result else None

    grid_path = args.out_dir / f"{timestamp}-grid.png"
    ocr_path = args.out_dir / f"{timestamp}-ocr.png"
    candidates_path = args.out_dir / f"{timestamp}-candidates.png"
    result_path = args.out_dir / f"{timestamp}-result.json"

    draw_grid(image, grid, grid_path)
    draw_ocr(image, lines, ocr_path)
    draw_candidates(image, candidates=candidates, selected=selected, output_path=candidates_path)

    payload = {
        "query": args.query,
        "source": str(source_path),
        "screen_size": list(image.size),
        "grid": {
            "cell_width": args.cell_width,
            "cell_height": args.cell_height,
            "taskbar_height": args.taskbar_height,
            "cell_count": len(grid),
        },
        "ocr_line_count": len(lines),
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
    print(f"selected=#{result.candidate.id} {result.candidate.label_text!r} score={result.candidate.score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
