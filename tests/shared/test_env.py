import os

from notepad_grounding.shared.env import load_env_file


def test_load_env_file_sets_missing_values_without_overwriting_existing(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=from-file\nOPENAI_MODEL='gpt-test'\nEXISTING=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("EXISTING", "already-set")

    load_env_file(env_path)

    assert os.environ["OPENAI_API_KEY"] == "from-file"
    assert os.environ["OPENAI_MODEL"] == "gpt-test"
    assert os.environ["EXISTING"] == "already-set"
