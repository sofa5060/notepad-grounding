from PIL import Image

from notepad_grounding.shared.grid_judge import GridJudgeResult
from notepad_grounding.shared.grid_judge import resolve_openai_judge_model


def test_grid_judge_result_accepts_true_false_verdicts():
    accepted = GridJudgeResult(
        contains_target=True,
        confidence=0.91,
        rationale="The Notepad icon is visible.",
        visible_evidence="Blue notepad icon and Notepad label",
    )
    rejected = GridJudgeResult(
        contains_target=False,
        confidence=0.87,
        rationale="The selected crop contains Steam, not Notepad.",
        visible_evidence="Steam label",
    )

    assert accepted.contains_target is True
    assert rejected.contains_target is False


def test_resolve_openai_judge_model_prefers_judge_model_then_openai_model(monkeypatch):
    monkeypatch.delenv("OPENAI_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert resolve_openai_judge_model() == "gpt-5.4"

    monkeypatch.setenv("OPENAI_MODEL", "chooser-model")
    assert resolve_openai_judge_model() == "chooser-model"

    monkeypatch.setenv("OPENAI_JUDGE_MODEL", "judge-model")
    assert resolve_openai_judge_model() == "judge-model"

    assert resolve_openai_judge_model("explicit") == "explicit"


class FakeGridJudge:
    def judge_crop(self, *, query: str, image: Image.Image) -> GridJudgeResult:
        return GridJudgeResult(
            contains_target=True,
            confidence=1.0,
            rationale=f"{query} is visible",
            visible_evidence=f"crop size {image.size}",
        )


def test_grid_judge_protocol_shape():
    result = FakeGridJudge().judge_crop(query="Notepad", image=Image.new("RGB", (10, 20)))

    assert result.contains_target is True
    assert result.visible_evidence == "crop size (10, 20)"
