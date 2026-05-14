from __future__ import annotations


def build_click_grid_prompt(*, query: str, cell_ids: list[str], rejected_cell_ids: list[str]) -> str:
    rejected = set(rejected_cell_ids)
    valid = [cell_id for cell_id in cell_ids if cell_id not in rejected]
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
        f"Valid cell_id values are: {', '.join(valid)}."
        f"{rejected_text} Do not return pixel coordinates."
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

