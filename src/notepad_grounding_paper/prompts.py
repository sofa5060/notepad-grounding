from __future__ import annotations


def build_cell_choice_prompt(*, query: str, cell_ids: list[str], style: str, rejected=(), reviewer_rationale: str = "") -> str:
    if style == "click":
        task = (
            "The image has a grid drawn over a screenshot crop, with column numbers above and row numbers "
            "to the left, outside the image content. Choose the single grid cell whose center should be "
            f"clicked to open the icon/app for: {query!r}. Focus on the icon graphic itself, not the text label."
        )
    else:
        task = f"Find the grid cell that most likely contains the desktop icon or shortcut for: {query!r}."
    prompt = f"You are helping a Windows desktop visual grounding system. {task}"
    if rejected:
        prompt += (
            f"\nDo NOT choose these rejected cells: {', '.join(rejected)}. "
            "A reviewer inspected them and said they do not contain the target."
        )
    if reviewer_rationale:
        prompt += f"\nReviewer rationale for the last rejection: {reviewer_rationale}"
    prompt += f"\nValid cell_id values are: {', '.join(cell_ids)}. Do not return pixel coordinates."
    return prompt


def build_target_review_prompt(*, query: str) -> str:
    return (
        "You are a strict reviewer for a Windows desktop visual grounding system.\n\n"
        f"Target query: {query!r}\n\n"
        "The image shows a grid of labeled cells drawn over a screenshot crop. Another model already "
        "selected one cell, which is HIGHLIGHTED with a thicker outline. Decide whether the HIGHLIGHTED "
        "cell actually contains the requested desktop item. Accept it if the highlighted cell contains either:\n"
        "1. recognizable visual evidence of the requested app/icon/shortcut, or\n"
        "2. visible label text matching the query.\n\n"
        "Look at the other cells only as context; judge only the highlighted one. Reject if the target is "
        "not visible in the highlighted cell, if it is in a different cell, or if the evidence is too "
        "ambiguous to continue safely."
    )


def build_desktop_review_prompt(*, action: str, expected: str) -> str:
    return (
        "You are a desktop automation reviewer. You validate whether an action succeeded "
        "by looking at the current screenshot.\n\n"
        f"Action just performed: {action}\n"
        f"Expected state: {expected}\n\n"
        "IMPORTANT: Only verify whether the expected app/window is OPEN on screen. "
        "Do NOT care whether it is the active/foreground window. "
        "As long as the app is visible somewhere on screen, report success.\n\n"
        "Determine:\n"
        "1. Is the expected app open and visible? (status: success)\n"
        "2. Is there an unexpected pop-up or wrong window open? (status: pop_up or wrong_app)\n"
        "3. If something is wrong, what is the exact recovery action? (action_needed)"
    )
