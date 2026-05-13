from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

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
        description="Vision-based Windows desktop icon grounding and automation.",
    )
    subparsers = parser.add_subparsers(dest="command")

    locate = subparsers.add_parser("locate", help="Locate a desktop icon by visible query.")
    locate.add_argument("--query", required=True, help="Visible target, for example Notepad.")
    locate.add_argument("--image", type=Path, default=None, help="Replay an existing screenshot instead of capturing.")

    automate = subparsers.add_parser("automate", help="Launch Notepad and save posts from JSONPlaceholder.")
    automate.add_argument("--query", required=True, help="Visible target, for example Notepad.")
    return parser


def run_locate(args: argparse.Namespace) -> int:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
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
        result = run_llm_visual_search(
            image,
            query=args.query,
            client=client,
            output_root=Path("output") / "llm_visual_search",
            rounds=3,
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


def run_automate(args: argparse.Namespace) -> int:
    try:
        client = OpenAIVisionClient()
        result = run_automation(
            query=args.query,
            client=client,
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

    print(f"flow=automate")
    print(f"output_dir={result.output_dir}")
    print(f"result={result.result_json}")
    print(f"total_posts={result.total_posts}")
    print(f"succeeded={result.succeeded}")
    print(f"failed={result.failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
