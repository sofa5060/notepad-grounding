from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from notepad_grounding.flow import run_flow


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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return start_flow(args)


def start_flow(args: argparse.Namespace) -> int:
    _setup_logging()
    try:
        result = run_flow(
            mode=args.command,
            query=args.query,
            output_root=Path("output"),
            max_retries=3,
            retry_delay=1.0,
            post_limit=10,
            llm_rounds=3,
        )

        if args.command == "locate":
            _print_locate_result(result)
            return 0

        _print_automation_result(result)
        return 0
    except Exception as exc:
        print(f"error: {exc}")
        return 3


def _print_automation_result(result) -> None:
    print("flow=automation")
    print(f"output_dir={result.output_dir}")
    print(f"result={result.result_json}")
    print(f"total_posts={result.total_posts}")
    print(f"succeeded={result.succeeded}")
    print(f"failed={result.failed}")


def _print_locate_result(result) -> None:
    print("flow=locate")
    print(f"output_dir={result.output_dir}")
    print(f"result={result.result_json}")
    print("found=true")
    print(f"center={result.center[0]},{result.center[1]}")


if __name__ == "__main__":
    raise SystemExit(main())
