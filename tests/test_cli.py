import pytest

from notepad_grounding.cli import build_parser


def test_default_parser_runs_automation_with_notepad_query():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.command == "run"
    assert args.query == "Notepad"


def test_default_parser_accepts_query_override():
    parser = build_parser()
    args = parser.parse_args(["--query", "Calculator"])

    assert args.command == "run"
    assert args.query == "Calculator"


def test_locate_parser_accepts_query_and_image():
    parser = build_parser()
    args = parser.parse_args(["locate", "--query", "Notepad", "--image", "screen.png"])

    assert args.command == "locate"
    assert args.query == "Notepad"
    assert str(args.image) == "screen.png"


def test_automate_subcommand_is_removed():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["automate", "--query", "Notepad"])
