import pytest

from notepad_grounding.shared.reviewer import ReviewResult
from notepad_grounding.shared.reviewer import parse_review_result


def test_parse_review_result_accepts_valid_json():
    result = parse_review_result(
        '{"status": "success", "action_needed": "proceed", "rationale": "Notepad is open"}'
    )

    assert result == ReviewResult(
        status="success",
        action_needed="proceed",
        rationale="Notepad is open",
    )


def test_parse_review_result_normalizes_status():
    result = parse_review_result(
        '{"status": "WRONG_APP", "action_needed": "close window", "rationale": "Steam opened"}'
    )

    assert result.status == "wrong_app"


def test_parse_review_result_defaults_unknown_status_to_error():
    result = parse_review_result(
        '{"status": "something_weird", "action_needed": "retry", "rationale": "?"}'
    )

    assert result.status == "error"
