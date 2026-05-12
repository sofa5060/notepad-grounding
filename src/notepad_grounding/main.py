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
from notepad_grounding.grounding.annotations import annotate_ocr_proof
from notepad_grounding.grounding.annotations import (
    annotate_screenshot,
    default_reference_boxes,
)
from notepad_grounding.grounding.candidates import infer_icon_candidates
from notepad_grounding.grounding.ocr import (
    OcrError,
    extract_ocr_lines,
    extract_ocr_words_tiled,
    extract_ocr_words,
    extract_windows_ocr_words,
    group_words_by_line,
    prepare_desktop_label_image,
    prepare_image_for_ocr,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "screenshot-proof":
        return run_screenshot_proof(args)
    if args.command == "candidate-proof":
        return run_candidate_proof(args)
    if args.command == "ocr-proof":
        return run_ocr_proof(args)

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
    candidate.add_argument(
        "--preprocess-mode",
        choices=("desktop_label", "standard"),
        default="desktop_label",
        help="OCR preprocessing mode. 'desktop_label' is optimized for icon text.",
    )

    ocr = subparsers.add_parser(
        "ocr-proof",
        help="Run OCR and save an annotated text-only proof screenshot.",
    )
    _add_image_source_arguments(ocr, output_help="Directory for OCR proof outputs.")
    ocr.add_argument(
        "--min-confidence",
        type=float,
        default=50,
        help="Minimum Tesseract word confidence to keep.",
    )
    ocr.add_argument(
        "--tesseract-cmd",
        type=str,
        default=None,
        help="Optional path to the Tesseract executable.",
    )
    ocr.add_argument(
        "--draw-words",
        action="store_true",
        help="Overlay raw word boxes in addition to grouped text boxes.",
    )
    ocr.add_argument(
        "--ocr-engine",
        choices=("windows", "tesseract"),
        default="windows",
        help="OCR backend to use. Windows OCR is preferred for GUI text.",
    )
    ocr.add_argument(
        "--ocr-mode",
        choices=("tiled", "full"),
        default="tiled",
        help="Tesseract-only mode: tiled OCR for small labels or full-screen OCR for comparison.",
    )
    ocr.add_argument(
        "--preprocess-mode",
        choices=("desktop_label", "standard"),
        default="desktop_label",
        help="OCR preprocessing mode. 'desktop_label' is optimized for icon text.",
    )
    ocr.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Save the preprocessed image that is fed to the OCR engine for debugging.",
    )
    ocr.add_argument(
        "--tile-size",
        type=int,
        default=360,
        help="Tile width and height for tiled OCR mode.",
    )
    ocr.add_argument(
        "--tile-overlap",
        type=int,
        default=80,
        help="Overlap in pixels between adjacent OCR tiles.",
    )
    ocr.add_argument(
        "--max-horizontal-gap",
        type=int,
        default=48,
        help="Maximum horizontal gap for combining OCR words on one line.",
    )
    ocr.add_argument(
        "--max-vertical-gap",
        type=int,
        default=6,
        help="Maximum vertical gap for merging wrapped desktop-label lines.",
    )
    ocr.add_argument(
        "--max-center-delta",
        type=int,
        default=48,
        help="Maximum center-x delta for merging wrapped desktop-label lines.",
    )

    return parser


def _add_image_source_arguments(
    parser: argparse.ArgumentParser,
    *,
    output_help: str,
) -> None:
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Existing screenshot to replay instead of capturing the live desktop.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=output_help,
    )
    parser.add_argument(
        "--expected-width",
        type=int,
        default=EXPECTED_SCREEN_SIZE[0],
        help="Expected Windows desktop width for metadata and optional validation.",
    )
    parser.add_argument(
        "--expected-height",
        type=int,
        default=EXPECTED_SCREEN_SIZE[1],
        help="Expected Windows desktop height for metadata and optional validation.",
    )
    parser.add_argument(
        "--strict-size",
        action="store_true",
        help="Exit with an error if the screenshot size differs from expected.",
    )
    parser.add_argument(
        "--allow-non-windows",
        action="store_true",
        help="Allow live capture attempts outside Windows for local smoke tests.",
    )


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
            preprocess_mode=args.preprocess_mode,
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


def run_ocr_proof(args: argparse.Namespace) -> int:
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

    # Save preprocessed debug image if requested
    if args.save_preprocessed:
        if args.preprocess_mode == "desktop_label":
            preprocessed = prepare_desktop_label_image(screenshot, upscale_factor=3)
        else:
            preprocessed = prepare_image_for_ocr(screenshot, upscale_factor=2)
        preprocessed_path = args.out_dir / f"{timestamp}-desktop-preprocessed.png"
        preprocessed.save(preprocessed_path)
        print(f"preprocessed_screenshot={preprocessed_path}")

    try:
        if args.ocr_engine == "windows":
            words = extract_windows_ocr_words(screenshot)
        elif args.ocr_mode == "tiled":
            words = extract_ocr_words_tiled(
                screenshot,
                min_confidence=args.min_confidence,
                tesseract_cmd=args.tesseract_cmd,
                tile_size=args.tile_size,
                overlap=args.tile_overlap,
                preprocess_mode=args.preprocess_mode,
            )
        else:
            words = extract_ocr_words(
                screenshot,
                min_confidence=args.min_confidence,
                tesseract_cmd=args.tesseract_cmd,
                preprocess_mode=args.preprocess_mode,
            )
    except OcrError as exc:
        print(f"error: {exc}")
        return 4

    lines = group_words_by_line(
        words,
        max_horizontal_gap=args.max_horizontal_gap,
        max_vertical_gap=args.max_vertical_gap,
        max_center_delta=args.max_center_delta,
    )
    annotated_path = args.out_dir / f"{timestamp}-desktop-ocr.png"
    notes = [
        f"runtime={platform.system()}",
        f"source={raw_path}",
        f"captured={actual_size[0]}x{actual_size[1]}",
        f"expected={expected_size[0]}x{expected_size[1]}",
        f"size_status={size_status}",
        f"ocr_words={len(words)}",
        f"ocr_groups={len(lines)}",
        f"ocr_engine={args.ocr_engine}",
        f"ocr_mode={args.ocr_mode}",
        f"preprocess_mode={args.preprocess_mode}",
        f"draw_words={args.draw_words}",
    ]
    annotate_ocr_proof(
        screenshot,
        words=words,
        lines=lines,
        output_path=annotated_path,
        draw_words=args.draw_words,
        notes=notes,
    )

    print(f"source_screenshot={raw_path}")
    print(f"ocr_screenshot={annotated_path}")
    print(f"captured_size={actual_size[0]}x{actual_size[1]}")
    print(f"ocr_words={len(words)}")
    print(f"ocr_groups={len(lines)}")
    print(f"ocr_engine={args.ocr_engine}")
    print(f"ocr_mode={args.ocr_mode}")
    print(f"preprocess_mode={args.preprocess_mode}")
    for index, line in enumerate(lines, start=1):
        print(
            f"ocr #{index}: "
            f"text={line.text!r} "
            f"box=({line.box.x1},{line.box.y1},{line.box.x2},{line.box.y2}) "
            f"confidence={line.confidence:.1f}"
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
