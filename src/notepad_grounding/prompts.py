from __future__ import annotations


def build_locate_prompt(*, query: str, width: int, height: int) -> str:
    return f"""
        You are looking at a Windows desktop screenshot.
        Dimensions: width={width}, height={height}. Coordinates start at the top-left.

        Locate this desktop icon/shortcut: {query}
        Follow the target description when deciding whether to use label text or visual icon appearance.

        Use x and y for the center of the icon graphic.
        Use bbox for only the visible icon graphic area, not the text label.
    """.strip()


def build_app_open_review_prompt(*, expected_app: str) -> str:
    return (
        f"Compare the desktop immediately before and after a double-click intended to open {expected_app}. "
        f"Set opened_expected_app to true only if the after screenshot shows {expected_app} newly opened. "
        "If another app opened or the evidence is uncertain, set it to false."
    )
