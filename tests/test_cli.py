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


def test_ocr_proof_help_exposes_grouping_controls(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["ocr-proof", "--help"])

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--draw-words" in help_text
    assert "--ocr-mode" in help_text
    assert "--tile-size" in help_text
    assert "--tile-overlap" in help_text
    assert "--max-horizontal-gap" in help_text
    assert "--max-vertical-gap" in help_text
