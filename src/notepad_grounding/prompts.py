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
    *,
    query: str,
    rejected_cell_ids: list[str],
    reviewer_rationale: str,
    valid_cell_ids: list[str],
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


def build_click_grid_prompt(*, query: str, cell_ids: list[str], rejected_cell_ids: list[str]) -> str:
    rejected = set(rejected_cell_ids)
    valid = [cell_id for cell_id in cell_ids if cell_id not in rejected]
    rejected_text = ""
    if rejected_cell_ids:
        rejected_text = (
            f" Do NOT choose rejected cells: {', '.join(rejected_cell_ids)}. "
            "A reviewer said those cells would miss the target."
        )
    return (
        "You are helping a Windows desktop visual grounding system choose a click target. "
        "The image has a yellow row/column grid drawn over a desktop screenshot crop. "
        "Column numbers are above, row numbers are on the left. "
        f"Pick the cell whose CENTER overlaps with the {query!r} icon graphic or label. "
        "If we click that center, it should open the app. "
        "Return JSON only with keys cell_id, confidence, rationale. "
        f"Valid cell_id values are: {', '.join(valid)}."
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
        "You are a reviewer for a Windows desktop visual grounding system.\n\n"
        f"Target: {query!r}\n\n"
        "The image shows a grid overlaid on a desktop screenshot. "
        "One cell is HIGHLIGHTED with a thick yellow outline. "
        "Its CENTER is the proposed mouse click point.\n\n"
        "Simple question: does the center of the yellow cell overlap with the "
        f"{query!r} icon graphic or its label? If we click there, will it open the app?\n\n"
        "If yes, accept. If the center would miss and click empty space or a different icon, reject."
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


def build_bbox_initial_prompt(*, query: str) -> str:
    return (
        "You are helping a Windows desktop visual grounding system. "
        f"Locate the icon GRAPHIC (the picture itself, NOT the text label) for: {query!r}. "
        "Return JSON only with keys: target_visible, icon_bbox, confidence, rationale. "
        "icon_bbox must be crop-local pixel coordinates [x1, y1, x2, y2]. "
        "The CENTER of icon_bbox will be used as the mouse click target, so align the box so its center "
        "lands on the visual center of the clickable icon graphic. "
        "Draw the box TIGHT around only the icon picture. It must NOT overlap other icons. "
        "Do not include the text label below the icon."
    )


def build_bbox_validation_prompt() -> str:
    return (
        "I drew your suggested bounding box in RED on the image above. "
        "Please review it carefully.\n\n"
        "The center point of this red rectangle will be used as the mouse click target. "
        "The red box must be aligned exactly on the clickable icon graphic so its center is not shifted "
        "too high, too low, left, or right.\n\n"
        "Is the red rectangle:\n"
        "1. Centered correctly over ONLY the icon graphic (not the text label)?\n"
        "2. Tight around the full clickable icon graphic, not clipped and not too loose?\n"
        "3. Not overlapping any neighboring icons or labels?\n\n"
        "Return JSON with keys: confirmed (true/false), corrected_icon_bbox, confidence, rationale.\n"
        "If confirmed=true, corrected_icon_bbox should be the same as before.\n"
        "If confirmed=false, give the corrected [x1, y1, x2, y2] coordinates."
    )
