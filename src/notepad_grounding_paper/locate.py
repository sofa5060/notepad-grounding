from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from notepad_grounding_paper.images import (
    Box,
    GridCell,
    build_grid_cells,
    cell_by_id,
    crop_around_point,
    draw_click_grid,
    draw_full_click_marker,
    draw_grid_cells,
    expand_box,
    offset_box,
    offset_point,
)
from notepad_grounding_paper.models import VisualSearchResult

logger = logging.getLogger(__name__)


def run_locate(
    image: Image.Image,
    *,
    query: str,
    client,
    output_root: Path,
    timestamp: str | None = None,
    rounds: int = 3,
    first_grid: tuple[int, int] = (3, 4),
    later_grid: tuple[int, int] = (3, 3),
    crop_padding: int = 40,
    final_crop_max_size: tuple[int, int] = (450, 350),
    target_reviewer=None,
    max_review_retries: int = 2,
) -> VisualSearchResult:
    """Find the screen coordinates of `query` (e.g. the Notepad icon) in `image`.

    Zooms in with LLM-guided grid choices, then picks the exact click point
    inside the final small crop.
    """
    start_time = time.perf_counter()
    run_id = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_dir / "00-source.png")

    bounds: Box = (0, 0, image.width, image.height)
    current_box = bounds

    # Zoom loop: draw a labeled grid, ask the LLM which cell holds the target,
    # shrink the region to that cell (+ padding), repeat until small enough.
    for round_index in range(1, rounds + 1):
        (width, height) = (current_box[2] - current_box[0], current_box[3] - current_box[1])
        if width <= final_crop_max_size[0] and height <= final_crop_max_size[1]:
            break

        crop = image.crop(current_box)
        (rows, cols) = first_grid if round_index == 1 else later_grid
        local_cells = build_grid_cells((0, 0, crop.width, crop.height), rows=rows, cols=cols, prefix=f"R{round_index}-")
        cell_ids = [cell.id for cell in local_cells]
        grid_image = draw_grid_cells(crop, local_cells, output_path=output_dir / f"{round_index:02d}-grid.png")
        choice = client.choose_cell(query=query, image=grid_image, cell_ids=cell_ids)
        logger.info("Round %d: chose cell %s (confidence %.2f)", round_index, choice.cell_id, choice.confidence)

        # Optional double-check: crop the chosen cell and let the reviewer
        # confirm the target is really there; on rejection ask the chooser
        # again with the rejected cells excluded.
        if target_reviewer is not None:
            rejected: list[str] = []
            for attempt_index in range(1, max_review_retries + 2):
                selected_box = offset_box(cell_by_id(local_cells, choice.cell_id).box, offset=current_box[:2])
                reviewed_crop = image.crop(expand_box(selected_box, padding=crop_padding, bounds=bounds))
                reviewed_crop.save(output_dir / f"{round_index:02d}-target-review-crop-attempt-{attempt_index}.png")
                review = target_reviewer.review_target_crop(query=query, image=reviewed_crop)
                if review.contains_target:
                    break
                rejected.append(choice.cell_id)
                logger.info("Round %d: reviewer rejected %s: %s", round_index, choice.cell_id, review.rationale)
                if attempt_index > max_review_retries:
                    raise ValueError(
                        f"Target reviewer rejected all attempts for round {round_index}; rejected={rejected}; last_rationale={review.rationale}"
                    )
                choice = client.revise_cell_choice(
                    query=query,
                    image=grid_image,
                    cell_ids=cell_ids,
                    rejected_cell_ids=rejected,
                    reviewer_rationale=review.rationale,
                    previous_response_id=choice.response_id,
                )
                logger.info("Round %d: revised to cell %s", round_index, choice.cell_id)

        # Map the winning cell to full-screen coordinates and zoom in.
        selected_box = offset_box(cell_by_id(local_cells, choice.cell_id).box, offset=current_box[:2])
        draw_grid_cells(
            crop,
            local_cells,
            output_path=output_dir / f"{round_index:02d}-selected.png",
            selected_cell_id=choice.cell_id,
        )
        current_box = expand_box(selected_box, padding=crop_padding, bounds=bounds)

    # Region is small enough: two-stage click-grid refinement → exact pixel.
    final_crop = image.crop(current_box)
    final_crop.save(output_dir / "final-crop.png")
    center = _run_marked_point_precision(
        final_crop,
        query=query,
        client=client,
        target_reviewer=target_reviewer,
        output_dir=output_dir,
        crop_box=current_box,
        screen_image=image,
    )
    return VisualSearchResult(
        query=query, center=center, output_dir=str(output_dir), elapsed_seconds=time.perf_counter() - start_time
    )


def _run_marked_point_precision(
    final_crop: Image.Image,
    *,
    query: str,
    client,
    target_reviewer,
    output_dir: Path,
    crop_box: Box,
    screen_image: Image.Image,
) -> tuple[int, int]:
    """Turn the final small crop into an exact screen click point: a coarse
    7x7 grid pass over the crop, then a fine 5x5 pass over a 96x96 patch
    around the coarse pick, mapped back to full-screen coordinates.
    """
    coarse_cell = _choose_reviewed_click_grid_cell(
        final_crop,
        query=query,
        client=client,
        target_reviewer=target_reviewer,
        cells=build_grid_cells((0, 0, final_crop.width, final_crop.height), rows=7, cols=7, id_fmt="rc"),
        overlay_path=output_dir / "click-points-01.png",
        output_dir=output_dir,
        artifact_prefix="click-grid-01",
        max_attempts=3,
    )

    (refinement_crop, refinement_crop_box) = crop_around_point(final_crop, center=coarse_cell.center, size=(96, 96))
    refinement_crop.save(output_dir / "click-points-02-crop.png")
    fine_cell = _choose_reviewed_click_grid_cell(
        refinement_crop,
        query=query,
        client=client,
        target_reviewer=target_reviewer,
        cells=build_grid_cells((0, 0, refinement_crop.width, refinement_crop.height), rows=5, cols=5, id_fmt="rc"),
        overlay_path=output_dir / "click-points-02.png",
        output_dir=output_dir,
        artifact_prefix="click-grid-02",
        max_attempts=3,
    )
    final_crop_point = offset_point(fine_cell.center, offset=refinement_crop_box[:2])
    screen_point = offset_point(final_crop_point, offset=crop_box[:2])

    # Marker images showing the final point on the crop and the full screen.
    marker_box = (final_crop_point[0] - 4, final_crop_point[1] - 4, final_crop_point[0] + 4, final_crop_point[1] + 4)
    marker = GridCell(id="CLICK", row=1, col=1, box=marker_box)
    draw_click_grid(final_crop, [marker], output_path=output_dir / "click-point-final.png", selected_cell_id="CLICK")
    draw_full_click_marker(screen_image, point=screen_point, output_path=output_dir / "click-point-full.png")

    return screen_point


def _choose_reviewed_click_grid_cell(
    image: Image.Image,
    *,
    query: str,
    client,
    target_reviewer,
    cells: list[GridCell],
    overlay_path: Path,
    output_dir: Path,
    artifact_prefix: str,
    max_attempts: int,
) -> GridCell:
    """One click-grid round: the LLM picks a cell on the numbered overlay;
    if a reviewer is set it must approve the highlighted pick, else retry.
    """
    overlay = draw_click_grid(image, cells, output_path=overlay_path)
    cell_ids = [cell.id for cell in cells]
    choice = client.choose_click_grid_cell(
        query=query, image=overlay, cell_ids=cell_ids, rejected_cell_ids=[], previous_response_id=None
    )

    if target_reviewer is None:
        draw_click_grid(image, cells, output_path=overlay_path, selected_cell_id=choice.cell_id)
        return cell_by_id(cells, choice.cell_id)

    rejected: list[str] = []
    for attempt_index in range(1, max_attempts + 1):
        # Re-draw with the pick highlighted and let the reviewer judge it.
        cell = cell_by_id(cells, choice.cell_id)
        selected_overlay = draw_click_grid(image, cells, output_path=overlay_path, selected_cell_id=choice.cell_id)
        selected_overlay.save(output_dir / f"{artifact_prefix}-target-review-crop-attempt-{attempt_index}.png")
        review = target_reviewer.review_target_grid_cell(query=query, image=selected_overlay)
        if review.contains_target:
            return cell
        rejected.append(choice.cell_id)
        logger.info("%s: reviewer rejected %s: %s", artifact_prefix, choice.cell_id, review.rationale)
        if attempt_index < max_attempts:
            choice = client.choose_click_grid_cell(
                query=query,
                image=overlay,
                cell_ids=cell_ids,
                rejected_cell_ids=rejected,
                previous_response_id=choice.response_id,
            )
            logger.info("%s: revised to cell %s", artifact_prefix, choice.cell_id)
    raise ValueError(
        f"Target reviewer rejected all attempts for {artifact_prefix}; rejected={rejected}; last_rationale={review.rationale}"
    )
