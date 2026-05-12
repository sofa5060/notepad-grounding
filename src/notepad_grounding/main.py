from __future__ import annotations

import argparse
import platform
from datetime import datetime
from pathlib import Path

from notepad_grounding.automation.desktop import (
    DesktopCaptureError,
    capture_desktop_screenshot,
)
from notepad_grounding.config import DEFAULT_OUTPUT_DIR, EXPECTED_SCREEN_SIZE
from notepad_grounding.grounding.annotations import (
    annotate_screenshot,
    default_reference_boxes,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "screenshot-proof":
        return run_screenshot_proof(args)

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


if __name__ == "__main__":
    raise SystemExit(main())
