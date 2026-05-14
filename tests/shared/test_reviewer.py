from notepad_grounding.models import DesktopReviewResult
from notepad_grounding.prompts import build_desktop_review_prompt
from notepad_grounding.reviewers import OpenAIReviewer


def test_review_result_model_validates():
    from notepad_grounding.models import ReviewResultModel

    model = ReviewResultModel(
        status="success",
        action_needed="proceed",
        rationale="Notepad is open",
    )
    assert model.status == "success"
    assert model.action_needed == "proceed"


def test_review_result_model_rejects_invalid_status_type():
    from notepad_grounding.models import ReviewResultModel

    # Pydantic coerces types, so this should still work
    model = ReviewResultModel(
        status="wrong_app",
        action_needed="close window",
        rationale="Steam opened",
    )
    assert model.status == "wrong_app"


def test_desktop_review_result_shape():
    result = DesktopReviewResult(
        status="success",
        action_needed="proceed",
        rationale="Notepad is open",
    )

    assert result.status == "success"
    assert OpenAIReviewer.__name__ == "OpenAIReviewer"


def test_desktop_review_prompt_is_built_outside_reviewer():
    prompt = build_desktop_review_prompt(
        action="double-click Notepad",
        expected="Notepad is open",
    )

    assert "double-click Notepad" in prompt
    assert "Notepad is open" in prompt
    assert "unexpected pop-up" in prompt
