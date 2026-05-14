from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from notepad_grounding.api import ApiError
from notepad_grounding.automate import run_automation
from notepad_grounding.capture import CaptureError
from notepad_grounding.capture import capture_desktop
from notepad_grounding.capture import load_image
from notepad_grounding.locate import run_locate
from notepad_grounding.reviewers import OpenAIBboxReviewer
from notepad_grounding.reviewers import OpenAIDesktopReviewer
from notepad_grounding.reviewers import OpenAITargetReviewer
from notepad_grounding.vision import OpenAIVisionClient


def _setup_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="notepad-grounding",
        description="Vision-based Windows desktop icon grounding and automation.",
    )
    parser.add_argument("--query", default="Notepad", help="Visible target query. Defaults to Notepad.")

    subparsers = parser.add_subparsers(dest="command", metavar="[command]")
    subparsers.default = "run"
    parser.set_defaults(command="run")
    locate = subparsers.add_parser("locate", help="Debug: locate a desktop icon without running automation.")
    locate.add_argument("--query", default="Notepad", help="Visible target query. Defaults to Notepad.")
    locate.add_argument("--image", type=Path, default=None, help="Replay an existing screenshot instead of capturing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "locate":
        return run_locate_command(args)
    return run_default_command(args)


def run_default_command(args: argparse.Namespace) -> int:
    _setup_logging()
    try:
        client = OpenAIVisionClient()
        result = run_automation(
            query=args.query,
            client=client,
            desktop_reviewer=OpenAIDesktopReviewer(),
            target_reviewer=OpenAITargetReviewer(),
            bbox_reviewer=OpenAIBboxReviewer(),
            output_root=Path("output"),
            max_retries=3,
            retry_delay=1.0,
            post_limit=10,
            llm_rounds=3,
        )
    except ApiError as exc:
        print(f"error: {exc}")
        return 4
    except Exception as exc:
        print(f"error: {exc}")
        return 3

    print("flow=automation")
    print(f"output_dir={result.output_dir}")
    print(f"result={result.result_json}")
    print(f"total_posts={result.total_posts}")
    print(f"succeeded={result.succeeded}")
    print(f"failed={result.failed}")
    return 0


def run_locate_command(args: argparse.Namespace) -> int:
    try:
        if args.image:
            image = load_image(args.image)
        else:
            image = capture_desktop()
    except (CaptureError, OSError) as exc:
        print(f"error: {exc}")
        return 2

    try:
        client = OpenAIVisionClient()
        result = run_locate(
            image,
            query=args.query,
            client=client,
            target_reviewer=OpenAITargetReviewer(),
            bbox_reviewer=OpenAIBboxReviewer(),
            output_root=Path("output") / "locate",
            rounds=3,
        )
    except Exception as exc:
        print(f"error: {exc}")
        return 3

    print("flow=locate")
    print(f"output_dir={result.output_dir}")
    print(f"result={result.result_json}")
    print("found=true")
    print(f"center={result.center[0]},{result.center[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
