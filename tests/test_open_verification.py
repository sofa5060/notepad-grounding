from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from notepad_grounding import automation
from notepad_grounding import llm


def test_verify_app_opened_sends_labeled_images_and_parses_boolean(monkeypatch):
    responses = MagicMock()
    responses.parse.return_value = SimpleNamespace(
        output_parsed=llm.AppOpenReview(opened_expected_app=True)
    )
    monkeypatch.setattr(llm, "OpenAI", lambda: SimpleNamespace(responses=responses))

    result = llm.verify_app_opened(
        expected_app="Windows Notepad",
        before_image=Image.new("RGB", (2, 2), "black"),
        after_image=Image.new("RGB", (2, 2), "white"),
    )

    assert result == llm.AppOpenReview(opened_expected_app=True)
    call = responses.parse.call_args.kwargs
    assert call["text_format"] is llm.AppOpenReview
    content = call["input"][0]["content"]
    assert content[1] == {"type": "input_text", "text": "Before screenshot:"}
    assert content[3] == {"type": "input_text", "text": "After screenshot:"}
    assert content[2]["image_url"].startswith("data:image/png;base64,")
    assert content[4]["image_url"].startswith("data:image/png;base64,")
    assert content[2]["image_url"] != content[4]["image_url"]


def test_verify_app_opened_returns_none_on_api_failure(monkeypatch):
    responses = MagicMock()
    responses.parse.side_effect = RuntimeError("offline")
    monkeypatch.setattr(llm, "OpenAI", lambda: SimpleNamespace(responses=responses))

    assert llm.verify_app_opened(
        expected_app="Windows Notepad",
        before_image=Image.new("RGB", (2, 2)),
        after_image=Image.new("RGB", (2, 2)),
    ) is None


def configure_click(monkeypatch, *, verdict, window_changes):
    before_image = Image.new("RGB", (2, 2), "black")
    after_image = Image.new("RGB", (2, 2), "white")
    monkeypatch.setattr(automation, "capture_desktop", MagicMock(side_effect=[before_image, after_image]))
    monkeypatch.setattr(automation, "verify_app_opened", MagicMock(return_value=verdict))
    monkeypatch.setattr(automation, "visible_window_handles", MagicMock(return_value={1}))
    monkeypatch.setattr(
        automation,
        "wait_for_visible_window_change",
        MagicMock(side_effect=window_changes),
    )
    monkeypatch.setattr(automation, "pyautogui", MagicMock())


def test_automate_post_verifies_before_typing_and_saves_artifacts(monkeypatch, tmp_path):
    configure_click(
        monkeypatch,
        verdict=llm.AppOpenReview(opened_expected_app=True),
        window_changes=[True],
    )
    type_content = MagicMock()
    save_file = MagicMock()
    close_notepad = MagicMock()
    monkeypatch.setattr(automation, "type_content", type_content)
    monkeypatch.setattr(automation, "save_file", save_file)
    monkeypatch.setattr(automation, "close_notepad", close_notepad)

    result = automation.automate_post(
        post={"id": 7, "title": "Title", "body": "Body"},
        index=1,
        target_dir=tmp_path,
        click_x=100,
        click_y=200,
        expected_app="Windows Notepad",
        output_dir=tmp_path / "run",
    )

    assert result == tmp_path / "post_7.txt"
    type_content.assert_called_once_with(content="Title: Title\n\nBody")
    save_file.assert_called_once_with(full_path=tmp_path / "post_7.txt")
    close_notepad.assert_called_once_with()
    assert (tmp_path / "run" / "open-before.png").exists()
    assert (tmp_path / "run" / "open-after.png").exists()
    assert (tmp_path / "run" / "open-verification.json").read_text() == (
        '{\n  "opened_expected_app": true\n}'
    )


@pytest.mark.parametrize(
    "opened_expected_app",
    [False, None],
    ids=["mismatch", "review-failure"],
)
def test_automate_post_closes_unverified_app_without_typing(monkeypatch, tmp_path, opened_expected_app):
    verdict = (
        llm.AppOpenReview(opened_expected_app=opened_expected_app)
        if opened_expected_app is not None
        else None
    )
    configure_click(monkeypatch, verdict=verdict, window_changes=[True, True])
    type_content = MagicMock()
    save_file = MagicMock()
    monkeypatch.setattr(automation, "type_content", type_content)
    monkeypatch.setattr(automation, "save_file", save_file)

    with pytest.raises(automation.AppOpenVerificationError):
        automation.automate_post(
            post={"id": 1, "title": "Title", "body": "Body"},
            index=1,
            target_dir=tmp_path,
            click_x=100,
            click_y=200,
            expected_app="Windows Notepad",
            output_dir=tmp_path / "run",
        )

    automation.pyautogui.hotkey.assert_called_once_with("alt", "f4")
    type_content.assert_not_called()
    save_file.assert_not_called()


def test_click_icon_aborts_when_unverified_app_does_not_close(monkeypatch, tmp_path):
    configure_click(
        monkeypatch,
        verdict=llm.AppOpenReview(opened_expected_app=False),
        window_changes=[True, False],
    )

    with pytest.raises(automation.AppCloseTimeoutError):
        automation.click_icon(
            x=100,
            y=200,
            expected_app="Windows Notepad",
            output_dir=tmp_path,
        )


def test_main_retries_recoverable_verification_failure(monkeypatch, tmp_path):
    from notepad_grounding import main

    coordinates = SimpleNamespace(x=100, y=200)
    expected_file = tmp_path / "post_1.txt"
    automate_post = MagicMock(
        side_effect=[automation.AppOpenVerificationError("wrong app"), expected_file]
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(main, "load_dotenv", MagicMock())
    monkeypatch.setattr(main, "reset_target_directory", MagicMock())
    monkeypatch.setattr(main, "fetch_posts", MagicMock(return_value=[{"id": 1}]))
    monkeypatch.setattr(main, "locate_icon", MagicMock(return_value=coordinates))
    monkeypatch.setattr(main, "automate_post", automate_post)

    main.main()

    assert automate_post.call_count == 2
    assert automate_post.call_args.kwargs["expected_app"] == "Windows Notepad"
    assert automate_post.call_args.kwargs["output_dir"] == tmp_path / "output/post_01/run_02"
