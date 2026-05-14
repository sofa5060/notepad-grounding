import pytest

from notepad_grounding import cli
from notepad_grounding.cli import build_parser


def test_main_delegates_to_start_flow(monkeypatch):
    seen = {}

    def fake_start_flow(args):
        seen["command"] = args.command
        seen["query"] = args.query
        return 17

    monkeypatch.setattr(cli, "start_flow", fake_start_flow)

    assert cli.main(["--query", "Calculator"]) == 17
    assert seen == {"command": "run", "query": "Calculator"}


def test_start_flow_calls_run_flow_for_default_command(monkeypatch, tmp_path):
    calls = {}

    class FakeResult:
        output_dir = str(tmp_path)
        result_json = str(tmp_path / "result.json")
        total_posts = 0
        succeeded = 0
        failed = 0

    def fake_run_flow(**kwargs):
        calls.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(cli, "run_flow", fake_run_flow)
    args = build_parser().parse_args(["--query", "Calculator"])

    assert cli.start_flow(args) == 0
    assert calls["mode"] == "run"
    assert calls["query"] == "Calculator"
    assert "dependencies" not in calls


def test_start_flow_calls_run_flow_for_locate_command(monkeypatch, tmp_path):
    calls = {}

    class FakeResult:
        output_dir = str(tmp_path)
        result_json = str(tmp_path / "result.json")
        center = (10, 20)

    def fake_run_flow(**kwargs):
        calls.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(cli, "run_flow", fake_run_flow)
    args = build_parser().parse_args(["locate", "--query", "Notepad"])

    assert cli.start_flow(args) == 0
    assert calls["mode"] == "locate"
    assert calls["query"] == "Notepad"
    assert "dependencies" not in calls


def test_start_flow_reports_runtime_errors_without_capture_error(monkeypatch, capsys):
    def fake_run_flow(**kwargs):
        raise RuntimeError("Live desktop capture must run inside Windows.")

    monkeypatch.setattr(cli, "run_flow", fake_run_flow)
    args = build_parser().parse_args([])

    assert cli.start_flow(args) == 3
    assert "Live desktop capture must run inside Windows." in capsys.readouterr().out


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


def test_locate_parser_accepts_query_only():
    parser = build_parser()
    args = parser.parse_args(["locate", "--query", "Notepad"])

    assert args.command == "locate"
    assert args.query == "Notepad"


def test_locate_parser_rejects_image_argument():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["locate", "--image", "screen.png"])


def test_automate_subcommand_is_removed():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["automate", "--query", "Notepad"])
