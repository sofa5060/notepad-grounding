from __future__ import annotations

from pathlib import Path

from direct_llm_grounding.simple_notepad_flow import filename_for_post
from direct_llm_grounding.simple_notepad_flow import format_post_content
from direct_llm_grounding.simple_notepad_flow import target_file_for_post


def test_format_post_content_matches_notepad_text() -> None:
    post = {"id": 7, "title": "A small title", "body": "Line one\nLine two"}

    assert format_post_content(post) == "Title: A small title\n\nLine one\nLine two"


def test_filename_for_post_uses_post_id() -> None:
    assert filename_for_post({"id": 12}, index=3) == "post_12.txt"


def test_filename_for_post_falls_back_to_index() -> None:
    assert filename_for_post({}, index=3) == "post_3.txt"


def test_target_file_for_post_joins_target_directory() -> None:
    assert target_file_for_post(Path("C:/Users/test/Desktop/tjm-project"), {"id": 4}, index=1) == Path(
        "C:/Users/test/Desktop/tjm-project/post_4.txt"
    )
