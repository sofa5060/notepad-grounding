import pytest

from notepad_grounding.main import build_parser


def test_candidate_proof_help_exposes_offline_and_tesseract_options(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["candidate-proof", "--help"])

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--image" in help_text
    assert "--tesseract-cmd" in help_text
