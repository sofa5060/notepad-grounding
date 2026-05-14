from notepad_grounding.models import TargetReviewResult
from notepad_grounding.prompts import build_target_review_prompt
from notepad_grounding.reviewers import resolve_openai_reviewer_model


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


def test_resolve_openai_reviewer_model_prefers_reviewer_then_legacy_judge_then_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_REVIEWER_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert resolve_openai_reviewer_model() == "gpt-5.4"

    monkeypatch.setenv("OPENAI_MODEL", "chooser-model")
    assert resolve_openai_reviewer_model() == "chooser-model"

    monkeypatch.setenv("OPENAI_JUDGE_MODEL", "judge-model")
    assert resolve_openai_reviewer_model() == "judge-model"

    monkeypatch.setenv("OPENAI_REVIEWER_MODEL", "reviewer-model")
    assert resolve_openai_reviewer_model() == "reviewer-model"

    assert resolve_openai_reviewer_model("explicit") == "explicit"


def test_target_review_prompt_is_built_outside_reviewer():
    prompt = build_target_review_prompt(query="Notepad")

    assert "Target query: 'Notepad'" in prompt
    assert "recognizable visual evidence" in prompt
    assert "visible label text" in prompt
