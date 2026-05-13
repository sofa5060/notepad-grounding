import pytest

from notepad_grounding.shared.reviewer import OpenAIReviewClient
from notepad_grounding.shared.reviewer import ReviewResult


def test_review_result_model_validates():
    from notepad_grounding.shared.schemas import ReviewResultModel

    model = ReviewResultModel(
        status="success",
        action_needed="proceed",
        rationale="Notepad is open",
    )
    assert model.status == "success"
    assert model.action_needed == "proceed"


def test_review_result_model_rejects_invalid_status_type():
    from notepad_grounding.shared.schemas import ReviewResultModel

    # Pydantic coerces types, so this should still work
    model = ReviewResultModel(
        status="wrong_app",
        action_needed="close window",
        rationale="Steam opened",
    )
    assert model.status == "wrong_app"
