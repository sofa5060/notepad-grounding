from notepad_grounding_paper.models import TargetReviewResult
from notepad_grounding_paper.prompts import build_target_review_prompt
from notepad_grounding_paper.reviewers import resolve_openai_reviewer_model


def test_target_review_result_accepts_true_false_verdicts():
    accepted = TargetReviewResult(
        contains_target=True,
        confidence=0.91,
        rationale="The Notepad icon is visible.",
        visible_evidence="Blue notepad icon and Notepad label",
    )
    rejected = TargetReviewResult(
        contains_target=False,
        confidence=0.87,
        rationale="The selected crop contains Steam, not Notepad.",
        visible_evidence="Steam label",
    )

    assert accepted.contains_target is True
    assert rejected.contains_target is False


def test_resolve_openai_reviewer_model_uses_openai_model(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert resolve_openai_reviewer_model() == "gpt-5.4"

    monkeypatch.setenv("OPENAI_MODEL", "chooser-model")
    assert resolve_openai_reviewer_model() == "chooser-model"

    assert resolve_openai_reviewer_model("explicit") == "explicit"


def test_target_review_prompt_is_built_outside_reviewer():
    prompt = build_target_review_prompt(query="Notepad")

    assert "Target query: 'Notepad'" in prompt
    assert "recognizable visual evidence" in prompt
    assert "visible label text" in prompt
