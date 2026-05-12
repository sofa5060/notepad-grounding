from __future__ import annotations

import argparse
import platform
from datetime import datetime
from pathlib import Path

from PIL import Image

from notepad_grounding.automation.desktop import (
    DesktopCaptureError,
    capture_desktop_screenshot,
)
from notepad_grounding.config import DEFAULT_OUTPUT_DIR, EXPECTED_SCREEN_SIZE
from notepad_grounding.grounding.annotations import annotate_candidate_proof
from notepad_grounding.grounding.annotations import (
    annotate_screenshot,
    default_reference_boxes,
)
from notepad_grounding.grounding.candidates import infer_icon_candidates
from notepad_grounding.grounding.ocr import OcrError, extract_ocr_lines


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "screenshot-proof":
        return run_screenshot_proof(args)
    if args.command == "candidate-proof":
        return run_candidate_proof(args)

    parser.print_help()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="notepad-grounding",
        description="Vision-based Windows desktop grounding proof.",
    )
    subparsers = parser.add_subparsers(dest="command")

    screenshot = subparsers.add_parser(
        "screenshot-proof",
        help="Capture the desktop and save raw plus annotated debug screenshots.",
    )
    screenshot.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for raw and annotated screenshot outputs.",
    )
    screenshot.add_argument(
        "--expected-width",
        type=int,
        default=EXPECTED_SCREEN_SIZE[0],
        help="Expected Windows desktop width for metadata and optional validation.",
    )
    screenshot.add_argument(
        "--expected-height",
        type=int,
        default=EXPECTED_SCREEN_SIZE[1],
        help="Expected Windows desktop height for metadata and optional validation.",
    )
    screenshot.add_argument(
        "--strict-size",
        action="store_true",
        help="Exit with an error if the captured screenshot size differs from expected.",
    )
    screenshot.add_argument(
        "--allow-non-windows",
        action="store_true",
        help="Allow capture attempts outside Windows for local smoke tests.",
    )

    candidate = subparsers.add_parser(
        "candidate-proof",
        help="Run OCR and save an annotated desktop icon candidate proof.",
    )
    candidate.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Existing screenshot to replay instead of capturing the live desktop.",
    )
    candidate.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for candidate proof outputs.",
    )
    candidate.add_argument(
        "--expected-width",
        type=int,
        default=EXPECTED_SCREEN_SIZE[0],
        help="Expected Windows desktop width for metadata and optional validation.",
    )
    candidate.add_argument(
        "--expected-height",
        type=int,
        default=EXPECTED_SCREEN_SIZE[1],
        help="Expected Windows desktop height for metadata and optional validation.",
    )
    candidate.add_argument(
        "--strict-size",
        action="store_true",
        help="Exit with an error if the screenshot size differs from expected.",
    )
    candidate.add_argument(
        "--allow-non-windows",
        action="store_true",
        help="Allow live capture attempts outside Windows for local smoke tests.",
    )
    candidate.add_argument(
        "--min-confidence",
        type=float,
        default=50,
        help="Minimum Tesseract word confidence to keep.",
    )
    candidate.add_argument(
        "--tesseract-cmd",
        type=str,
        default=None,
        help="Optional path to the Tesseract executable.",
    )

    return parser


def run_screenshot_proof(args: argparse.Namespace) -> int:
    try:
        screenshot = capture_desktop_screenshot(
            require_windows=not args.allow_non_windows,
        )
    except DesktopCaptureError as exc:
        print(f"error: {exc}")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_path = args.out_dir / f"{timestamp}-desktop-raw.png"
    annotated_path = args.out_dir / f"{timestamp}-desktop-annotated.png"

    screenshot.save(raw_path)

    actual_size = screenshot.size
    expected_size = (args.expected_width, args.expected_height)
    size_status = "ok" if actual_size == expected_size else "mismatch"
    notes = [
        f"runtime={platform.system()}",
        f"captured={actual_size[0]}x{actual_size[1]}",
        f"expected={expected_size[0]}x{expected_size[1]}",
        f"size_status={size_status}",
    ]
    annotate_screenshot(
        screenshot,
        boxes=default_reference_boxes(*actual_size),
        output_path=annotated_path,
        title="Milestone 2 screenshot coordinate proof",
        notes=notes,
    )

    print(f"raw_screenshot={raw_path}")
    print(f"annotated_screenshot={annotated_path}")
    print(f"captured_size={actual_size[0]}x{actual_size[1]}")

    if actual_size != expected_size:
        message = (
            f"warning: expected {expected_size[0]}x{expected_size[1]} but captured "
            f"{actual_size[0]}x{actual_size[1]}"
        )
        print(message)
        if args.strict_size:
            return 3

    return 0


def run_candidate_proof(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        screenshot, raw_path = _load_or_capture_candidate_image(args, timestamp)
    except DesktopCaptureError as exc:
        print(f"error: {exc}")
        return 2
    except OSError as exc:
        print(f"error: unable to load screenshot image: {exc}")
        return 2

    actual_size = screenshot.size
    expected_size = (args.expected_width, args.expected_height)
    size_status = "ok" if actual_size == expected_size else "mismatch"

    try:
        labels = extract_ocr_lines(
            screenshot,
            min_confidence=args.min_confidence,
            tesseract_cmd=args.tesseract_cmd,
        )
    except OcrError as exc:
        print(f"error: {exc}")
        return 4

    candidates = infer_icon_candidates(labels, screen_size=actual_size)
    annotated_path = args.out_dir / f"{timestamp}-desktop-candidates.png"
    notes = [
        f"runtime={platform.system()}",
        f"source={raw_path}",
        f"captured={actual_size[0]}x{actual_size[1]}",
        f"expected={expected_size[0]}x{expected_size[1]}",
        f"size_status={size_status}",
        f"ocr_labels={len(labels)}",
        f"candidates={len(candidates)}",
    ]
    annotate_candidate_proof(
        screenshot,
        labels=labels,
        candidates=candidates,
        output_path=annotated_path,
        notes=notes,
    )

    print(f"source_screenshot={raw_path}")
    print(f"candidate_screenshot={annotated_path}")
    print(f"captured_size={actual_size[0]}x{actual_size[1]}")
    print(f"ocr_labels={len(labels)}")
    print(f"candidates={len(candidates)}")
    for candidate in candidates:
        print(
            f"candidate #{candidate.candidate_id}: "
            f"text={candidate.label_text!r} "
            f"label=({candidate.label_box.x1},{candidate.label_box.y1},"
            f"{candidate.label_box.x2},{candidate.label_box.y2}) "
            f"icon=({candidate.icon_box.x1},{candidate.icon_box.y1},"
            f"{candidate.icon_box.x2},{candidate.icon_box.y2})"
        )

    if actual_size != expected_size:
        message = (
            f"warning: expected {expected_size[0]}x{expected_size[1]} but loaded "
            f"{actual_size[0]}x{actual_size[1]}"
        )
        print(message)
        if args.strict_size:
            return 3

    return 0


def _load_or_capture_candidate_image(
    args: argparse.Namespace,
    timestamp: str,
) -> tuple[Image.Image, Path]:
    if args.image:
        image = Image.open(args.image).convert("RGB")
        return image, args.image

    screenshot = capture_desktop_screenshot(require_windows=not args.allow_non_windows)
    raw_path = args.out_dir / f"{timestamp}-desktop-raw.png"
    screenshot.save(raw_path)
    return screenshot, raw_path


if __name__ == "__main__":
    raise SystemExit(main())
