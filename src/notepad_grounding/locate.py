from __future__ import annotations

import json
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from notepad_grounding.click_grid import ClickGridCell
from notepad_grounding.click_grid import build_click_grid_cells
from notepad_grounding.click_grid import crop_around_point
from notepad_grounding.click_grid import draw_click_grid
from notepad_grounding.click_grid import draw_full_click_marker
from notepad_grounding.click_grid import offset_point
from notepad_grounding.click_grid import grid_cell_by_id
from notepad_grounding.geometry import Box
from notepad_grounding.geometry import build_grid_cells
from notepad_grounding.geometry import clamp_box
from notepad_grounding.geometry import expand_box
from notepad_grounding.reviewers import BboxReviewer
from notepad_grounding.reviewers import TargetReviewer
from notepad_grounding.images import crop_box
from notepad_grounding.images import draw_box
from notepad_grounding.images import draw_grid_cells
from notepad_grounding.vision import VisionClient


@dataclass(frozen=True)
class TargetReviewAttempt:
    attempt_index: int
    selected_cell_id: str
    selected_box: Box
    reviewed_crop_box: Box
    crop_image: str
    contains_target: bool
    confidence: float
    rationale: str
    visible_evidence: str


@dataclass(frozen=True)
class VisualSearchStep:
    round_index: int
    crop_box: Box
    selected_cell_id: str
    selected_box: Box
    confidence: float
    rationale: str
    grid_image: str
    review_attempts: list[TargetReviewAttempt]


@dataclass(frozen=True)
class FinalDetectionStep:
    crop_box: Box
    icon_bbox_local: Box
    icon_bbox_screen: Box
    confidence: float
    rationale: str
    crop_image: str
    detection_image: str


@dataclass(frozen=True)
class FinalClickPointStep:
    crop_box: Box
    coarse_point_id: str
    fine_point_id: str
    crop_point: tuple[int, int]
    screen_point: tuple[int, int]
    first_overlay_image: str
    first_result_json: str
    refinement_crop_box: Box
    refinement_crop_image: str
    second_overlay_image: str
    second_result_json: str
    final_image: str
    full_image: str
    result_json: str
    confidence: float
    rationale: str


@dataclass(frozen=True)
class VisualSearchResult:
    query: str
    center: tuple[int, int]
    final_box: Box
    steps: list[VisualSearchStep]
    final_method: str
    final_detection: FinalDetectionStep | None
    final_click_point: FinalClickPointStep | None
    output_dir: str
    result_json: str
    elapsed_seconds: float


def run_locate(
    image: Image.Image,
    *,
    query: str,
    client: VisionClient,
    output_root: Path,
    timestamp: str | None = None,
    rounds: int = 3,
    first_grid: tuple[int, int] = (3, 4),
    later_grid: tuple[int, int] = (3, 3),
    crop_padding: int = 40,
    final_crop_max_size: tuple[int, int] = (450, 350),
    target_reviewer: TargetReviewer | None = None,
    bbox_reviewer: BboxReviewer | None = None,
    max_review_retries: int = 2,
    final_precision: str = "marked-point",
    bbox_fallback: bool = True,
) -> VisualSearchResult:
    start_time = time.perf_counter()
    run_id = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "00-source.png"
    image.convert("RGB").save(source_path)

    bounds: Box = (0, 0, image.width, image.height)
    current_box = bounds
    steps: list[VisualSearchStep] = []
    selected_box = bounds

    for round_index in range(1, rounds + 1):
        if _crop_is_small_enough(current_box, max_size=final_crop_max_size):
            break

        crop = crop_box(image, current_box)
        rows, cols = first_grid if round_index == 1 else later_grid
        prefix = f"R{round_index}-"
        local_cells = build_grid_cells(
            (0, 0, crop.width, crop.height),
            rows=rows,
            cols=cols,
            prefix=prefix,
        )
        grid_path = output_dir / f"{round_index:02d}-grid.png"
        draw_grid_cells(crop, local_cells, output_path=grid_path)

        choice = client.choose_cell(
            query=query,
            image=Image.open(grid_path).convert("RGB"),
            cell_ids=[cell.id for cell in local_cells],
        )

        review_attempts: list[TargetReviewAttempt] = []
        rejected_cell_ids: list[str] = []
        selected_local = _cell_by_id(local_cells, choice.cell_id)
        selected_box = _offset_box(selected_local.box, offset_x=current_box[0], offset_y=current_box[1])

        if target_reviewer is not None:
            for attempt_index in range(1, max_review_retries + 2):
                selected_local = _cell_by_id(local_cells, choice.cell_id)
                selected_box = _offset_box(selected_local.box, offset_x=current_box[0], offset_y=current_box[1])
                reviewed_crop_box = expand_box(selected_box, padding=crop_padding, bounds=bounds)
                reviewed_crop = crop_box(image, reviewed_crop_box)
                review_crop_path = output_dir / f"{round_index:02d}-target-review-crop-attempt-{attempt_index}.png"
                reviewed_crop.save(review_crop_path)

                review_result = target_reviewer.review_target_crop(query=query, image=reviewed_crop)
                review_result_path = output_dir / f"{round_index:02d}-target-review-result-attempt-{attempt_index}.json"
                attempt = TargetReviewAttempt(
                    attempt_index=attempt_index,
                    selected_cell_id=choice.cell_id,
                    selected_box=selected_box,
                    reviewed_crop_box=reviewed_crop_box,
                    crop_image=str(review_crop_path),
                    contains_target=review_result.contains_target,
                    confidence=review_result.confidence,
                    rationale=review_result.rationale,
                    visible_evidence=review_result.visible_evidence,
                )
                review_result_path.write_text(json.dumps(asdict(attempt), indent=2), encoding="utf-8")
                review_attempts.append(attempt)

                if review_result.contains_target:
                    break

                rejected_cell_ids.append(choice.cell_id)
                if attempt_index > max_review_retries:
                    raise ValueError(
                        "Target reviewer rejected all attempts "
                        f"for round {round_index}; rejected={rejected_cell_ids}; "
                        f"last_rationale={review_result.rationale}"
                    )

                choice = client.revise_cell_choice(
                    query=query,
                    image=Image.open(grid_path).convert("RGB"),
                    cell_ids=[cell.id for cell in local_cells],
                    rejected_cell_ids=rejected_cell_ids,
                    reviewer_rationale=review_result.rationale,
                    previous_response_id=choice.response_id,
                )

        selected_grid_path = output_dir / f"{round_index:02d}-selected.png"
        draw_grid_cells(
            crop,
            local_cells,
            output_path=selected_grid_path,
            selected_cell_id=choice.cell_id,
        )

        steps.append(
            VisualSearchStep(
                round_index=round_index,
                crop_box=current_box,
                selected_cell_id=choice.cell_id,
                selected_box=selected_box,
                confidence=choice.confidence,
                rationale=choice.rationale,
                grid_image=str(selected_grid_path),
                review_attempts=review_attempts,
            )
        )
        current_box = expand_box(selected_box, padding=crop_padding, bounds=bounds)

    # Final step: iterative bbox refinement with validation
    final_crop = crop_box(image, current_box)
    final_crop_path = output_dir / "final-crop.png"
    final_crop.save(final_crop_path)

    final_method = "marked_point"
    final_detection: FinalDetectionStep | None = None
    final_click_point: FinalClickPointStep | None = None

    if final_precision == "marked-point":
        try:
            final_click_point = _run_marked_point_precision(
                final_crop,
                query=query,
                client=client,
                target_reviewer=target_reviewer,
                output_dir=output_dir,
                crop_box=current_box,
                screen_image=image,
                bounds=bounds,
            )
            center = final_click_point.screen_point
            final_box = _point_box(center, bounds=bounds)
        except Exception as exc:
            if not bbox_fallback:
                raise
            (output_dir / "click-point-error.json").write_text(
                json.dumps({"error": str(exc)}, indent=2),
                encoding="utf-8",
            )
            final_method = "bbox_fallback"
            center, final_box, final_detection = _run_bbox_precision(
                final_crop,
                query=query,
                bbox_reviewer=bbox_reviewer,
                output_dir=output_dir,
                crop_box=current_box,
                screen_image=image,
                bounds=bounds,
                final_crop_path=final_crop_path,
            )
    elif final_precision == "bbox":
        final_method = "bbox"
        center, final_box, final_detection = _run_bbox_precision(
            final_crop,
            query=query,
            bbox_reviewer=bbox_reviewer,
            output_dir=output_dir,
            crop_box=current_box,
            screen_image=image,
            bounds=bounds,
            final_crop_path=final_crop_path,
        )
    else:
        raise ValueError(f"Unknown final_precision: {final_precision}")

    elapsed = time.perf_counter() - start_time
    print(f"[TIMING] Visual search completed in {elapsed:.2f} seconds")

    result_path = output_dir / "result.json"
    result = VisualSearchResult(
        query=query,
        center=center,
        final_box=final_box,
        steps=steps,
        final_method=final_method,
        final_detection=final_detection,
        final_click_point=final_click_point,
        output_dir=str(output_dir),
        result_json=str(result_path),
        elapsed_seconds=elapsed,
    )
    result_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def _cell_by_id(cells, cell_id: str):
    for cell in cells:
        if cell.id == cell_id:
            return cell
    raise ValueError(f"Unknown selected cell: {cell_id}")


def _offset_box(box: Box, *, offset_x: int, offset_y: int) -> Box:
    return (
        box[0] + offset_x,
        box[1] + offset_y,
        box[2] + offset_x,
        box[3] + offset_y,
    )


def _run_marked_point_precision(
    final_crop: Image.Image,
    *,
    query: str,
    client: VisionClient,
    target_reviewer: TargetReviewer | None,
    output_dir: Path,
    crop_box: Box,
    screen_image: Image.Image,
    bounds: Box,
) -> FinalClickPointStep:
    coarse_cells = build_click_grid_cells(final_crop.size, rows=7, cols=7)
    first_overlay_path = output_dir / "click-points-01.png"
    first_choice, coarse_cell = _choose_reviewed_click_grid_cell(
        final_crop,
        query=query,
        client=client,
        target_reviewer=target_reviewer,
        cells=coarse_cells,
        overlay_path=first_overlay_path,
        output_dir=output_dir,
        artifact_prefix="click-grid-01",
        max_attempts=3,
    )
    first_result_path = output_dir / "click-points-01-result.json"
    first_result_path.write_text(
        json.dumps(
            {
                "cell_id": first_choice.cell_id,
                "confidence": first_choice.confidence,
                "rationale": first_choice.rationale,
                "response_id": first_choice.response_id,
                "crop_point": coarse_cell.center,
                "cell_box": coarse_cell.box,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    refinement_crop, refinement_crop_box = crop_around_point(final_crop, center=coarse_cell.center, size=(96, 96))
    refinement_crop_path = output_dir / "click-points-02-crop.png"
    refinement_crop.save(refinement_crop_path)

    fine_cells = build_click_grid_cells(refinement_crop.size, rows=5, cols=5)
    second_overlay_path = output_dir / "click-points-02.png"
    second_choice, fine_cell = _choose_reviewed_click_grid_cell(
        refinement_crop,
        query=query,
        client=client,
        target_reviewer=target_reviewer,
        cells=fine_cells,
        overlay_path=second_overlay_path,
        output_dir=output_dir,
        artifact_prefix="click-grid-02",
        max_attempts=3,
    )
    final_crop_point = offset_point(fine_cell.center, offset=(refinement_crop_box[0], refinement_crop_box[1]))
    screen_point = offset_point(final_crop_point, offset=(crop_box[0], crop_box[1]))

    second_result_path = output_dir / "click-points-02-result.json"
    second_result_path.write_text(
        json.dumps(
            {
                "cell_id": second_choice.cell_id,
                "confidence": second_choice.confidence,
                "rationale": second_choice.rationale,
                "response_id": second_choice.response_id,
                "refinement_crop_point": fine_cell.center,
                "final_crop_point": final_crop_point,
                "screen_point": screen_point,
                "cell_box": fine_cell.box,
                "refinement_crop_box": refinement_crop_box,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    final_image_path = output_dir / "click-point-final.png"
    final_cell = ClickGridCell(
        id="CLICK",
        row=1,
        col=1,
        box=(final_crop_point[0] - 4, final_crop_point[1] - 4, final_crop_point[0] + 4, final_crop_point[1] + 4),
    )
    draw_click_grid(
        final_crop,
        [final_cell],
        output_path=final_image_path,
        selected_cell_id="CLICK",
    )
    full_image_path = output_dir / "click-point-full.png"
    draw_full_click_marker(
        screen_image,
        point=screen_point,
        output_path=full_image_path,
        label="click_point",
    )
    result_path = output_dir / "click-point-final.json"
    result = FinalClickPointStep(
        crop_box=crop_box,
        coarse_point_id=first_choice.cell_id,
        fine_point_id=second_choice.cell_id,
        crop_point=final_crop_point,
        screen_point=screen_point,
        first_overlay_image=str(first_overlay_path),
        first_result_json=str(first_result_path),
        refinement_crop_box=refinement_crop_box,
        refinement_crop_image=str(refinement_crop_path),
        second_overlay_image=str(second_overlay_path),
        second_result_json=str(second_result_path),
        final_image=str(final_image_path),
        full_image=str(full_image_path),
        result_json=str(result_path),
        confidence=min(first_choice.confidence, second_choice.confidence),
        rationale=f"{first_choice.rationale} | {second_choice.rationale}",
    )
    result_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def _choose_reviewed_click_grid_cell(
    image: Image.Image,
    *,
    query: str,
    client: VisionClient,
    target_reviewer: TargetReviewer | None,
    cells: list[ClickGridCell],
    overlay_path: Path,
    output_dir: Path,
    artifact_prefix: str,
    max_attempts: int,
) -> tuple[object, ClickGridCell]:
    rejected_cell_ids: list[str] = []
    previous_response_id: str | None = None
    last_rationale = ""

    for attempt_index in range(1, max_attempts + 1):
        draw_click_grid(image, cells, output_path=overlay_path)
        choice = client.choose_click_grid_cell(
            query=query,
            image=Image.open(overlay_path).convert("RGB"),
            cell_ids=[cell.id for cell in cells],
            rejected_cell_ids=rejected_cell_ids,
            previous_response_id=previous_response_id,
        )
        previous_response_id = choice.response_id
        cell = grid_cell_by_id(cells, choice.cell_id)
        draw_click_grid(image, cells, output_path=overlay_path, selected_cell_id=choice.cell_id)

        if target_reviewer is None:
            return choice, cell

        reviewed_crop_box = expand_box(cell.box, padding=12, bounds=(0, 0, image.width, image.height))
        reviewed_crop = image.crop(reviewed_crop_box)
        review_crop_path = output_dir / f"{artifact_prefix}-target-review-crop-attempt-{attempt_index}.png"
        reviewed_crop.save(review_crop_path)
        review_result = target_reviewer.review_target_crop(query=query, image=reviewed_crop)
        review_result_path = output_dir / f"{artifact_prefix}-target-review-result-attempt-{attempt_index}.json"
        review_result_path.write_text(
            json.dumps(
                {
                    "attempt_index": attempt_index,
                    "selected_cell_id": choice.cell_id,
                    "cell_box": cell.box,
                    "reviewed_crop_box": reviewed_crop_box,
                    "crop_image": str(review_crop_path),
                    "contains_target": review_result.contains_target,
                    "confidence": review_result.confidence,
                    "rationale": review_result.rationale,
                    "visible_evidence": review_result.visible_evidence,
                    "response_id": choice.response_id,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if review_result.contains_target:
            return choice, cell

        rejected_cell_ids.append(choice.cell_id)
        last_rationale = review_result.rationale

    raise ValueError(
        f"Click grid target reviewer rejected all attempts for {artifact_prefix}; "
        f"rejected={rejected_cell_ids}; last_rationale={last_rationale}"
    )


def _run_bbox_precision(
    final_crop: Image.Image,
    *,
    query: str,
    bbox_reviewer: BboxReviewer | None,
    output_dir: Path,
    crop_box: Box,
    screen_image: Image.Image,
    bounds: Box,
    final_crop_path: Path,
) -> tuple[tuple[int, int], Box, FinalDetectionStep]:
    if bbox_reviewer is None:
        raise ValueError("bbox_reviewer is required for bbox precision")
    detection = bbox_reviewer.review_bbox(query=query, image=final_crop, debug_dir=output_dir)
    if not detection.target_visible:
        raise ValueError(f"LLM reported target not visible in final crop: {detection.rationale}")

    icon_box = _offset_box(detection.icon_bbox, offset_x=crop_box[0], offset_y=crop_box[1])
    icon_box = clamp_box(icon_box, bounds)
    final_detection_path = output_dir / "final-detection.png"
    draw_box(final_crop, detection.icon_bbox, output_path=final_detection_path, label="icon_bbox")

    full_detection_path = output_dir / "full-detection.png"
    draw_box(screen_image, icon_box, output_path=full_detection_path, label="icon_bbox")

    center = ((icon_box[0] + icon_box[2]) // 2, (icon_box[1] + icon_box[3]) // 2)
    final_detection = FinalDetectionStep(
        crop_box=crop_box,
        icon_bbox_local=detection.icon_bbox,
        icon_bbox_screen=icon_box,
        confidence=detection.confidence,
        rationale=detection.rationale,
        crop_image=str(final_crop_path),
        detection_image=str(final_detection_path),
    )
    return center, icon_box, final_detection


def _point_box(point: tuple[int, int], *, bounds: Box) -> Box:
    return clamp_box((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), bounds)


def _crop_is_small_enough(box: Box, *, max_size: tuple[int, int]) -> bool:
    return (box[2] - box[0]) <= max_size[0] and (box[3] - box[1]) <= max_size[1]
