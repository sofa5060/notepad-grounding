from __future__ import annotations


def build_cell_choice_prompt(*, query: str, cell_ids: list[str]) -> str:
    return (
        "You are helping a Windows desktop visual grounding system. "
        f"Find the grid cell that most likely contains the desktop icon or shortcut for: {query!r}. "
        "Return JSON only with keys cell_id, confidence, rationale. "
        f"Valid cell_id values are: {', '.join(cell_ids)}. "
        "Do not return pixel coordinates."
    )


def build_revise_cell_choice_prompt(
    *, query: str, rejected_cell_ids: list[str], reviewer_rationale: str, valid_cell_ids: list[str]
) -> str:
    return (
        "You are helping a Windows desktop visual grounding system. "
        f"You previously selected grid cell(s) {', '.join(rejected_cell_ids)} for target {query!r}, "
        "but a reviewer inspected the selected crop and rejected it because it did not contain the target.\n\n"
        f"Reviewer rationale: {reviewer_rationale}\n\n"
        f"Choose a different grid cell that most likely contains the desktop icon or shortcut for {query!r}. "
        f"Do NOT choose any rejected cell: {', '.join(rejected_cell_ids)}. "
        "Return JSON only with keys cell_id, confidence, rationale. "
        f"Valid cell_id values are: {', '.join(valid_cell_ids)}. "
        "Do not return pixel coordinates."
    )


def build_click_grid_prompt(*, query: str, valid_cell_ids: list[str], rejected_cell_ids: list[str]) -> str:
    rejected_text = ""
    if rejected_cell_ids:
        rejected_text = (
            f" Do NOT choose rejected cells: {', '.join(rejected_cell_ids)}. "
            "A reviewer inspected those cells and said they do not contain the target."
        )
    return (
        "You are helping a Windows desktop visual grounding system choose a click target. "
        "The image has a yellow row/column grid drawn over the screenshot crop. "
        "Column numbers are shown above the image and row numbers are shown on the left, outside the image content. "
        f"Choose the single grid cell whose center should be clicked to open the icon/app for: {query!r}. "
        "Focus on the icon graphic itself, not the text label. "
        "Return JSON only with keys cell_id, confidence, rationale, where cell_id uses the format R<row>C<column>, for example R3C4. "
        f"Valid cell_id values are: {', '.join(valid_cell_ids)}."
        f"{rejected_text} Do not return pixel coordinates."
    )


def build_choice_correction_prompt(*, error: str, valid_cell_ids: list[str]) -> str:
    return (
        "Your previous response could not be used.\n"
        f"Problem: {error}\n"
        f"Valid cell_id values are: {', '.join(valid_cell_ids)}.\n"
        "Return JSON only with keys cell_id, confidence, rationale using one valid cell_id. "
        "Do not return pixel coordinates."
    )


def build_target_review_prompt(*, query: str) -> str:
    return (
        "You are a strict reviewer for a Windows desktop visual grounding system.\n\n"
        f"Target query: {query!r}\n\n"
        "The image is a crop from a selected grid cell. Decide whether this crop contains "
        "the requested desktop item. Accept the crop if it contains either:\n"
        "1. recognizable visual evidence of the requested app/icon/shortcut, or\n"
        "2. visible label text matching the query.\n\n"
        "Reject the crop if the target is not visible, if it only contains a different app, "
        "or if the evidence is too ambiguous to continue safely."
    )


def build_target_grid_review_prompt(*, query: str) -> str:
    return (
        "You are a strict reviewer for a Windows desktop visual grounding system.\n\n"
        f"Target query: {query!r}\n\n"
        "The image shows a grid of labeled cells drawn over a screenshot crop. "
        "Another model already selected one cell, which is HIGHLIGHTED with a thicker yellow outline. "
        "Your job: decide whether the HIGHLIGHTED cell actually contains the requested desktop item. "
        "Accept it if the highlighted cell contains either:\n"
        "1. recognizable visual evidence of the requested app/icon/shortcut, or\n"
        "2. visible label text matching the query.\n\n"
        "Look at the OTHER cells too as context, but ONLY judge the highlighted one. "
        "Reject if the target is not visible in the highlighted cell, "
        "if it is in a DIFFERENT cell, or if the evidence is too ambiguous to continue safely."
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
        "1. Is the expected app open and visible?\n"
        "2. Is there an unexpected pop-up or wrong window open?\n"
        "3. If something is wrong, what is the exact recovery action?"
    )
