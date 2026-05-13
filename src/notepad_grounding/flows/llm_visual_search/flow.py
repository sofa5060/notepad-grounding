from __future__ import annotations

import json
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from notepad_grounding.shared.geometry import Box
from notepad_grounding.shared.geometry import build_grid_cells
from notepad_grounding.shared.geometry import expand_box
from notepad_grounding.shared.grid_judge import GridJudgeClient
from notepad_grounding.shared.images import crop_box
from notepad_grounding.shared.images import draw_box
from notepad_grounding.shared.images import draw_grid_cells
from notepad_grounding.shared.llm import VisionClient


@dataclass(frozen=True)
class GridJudgeAttempt:
    attempt_index: int
    selected_cell_id: str
    selected_box: Box
    judged_crop_box: Box
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
    judge_attempts: list[GridJudgeAttempt]


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
class VisualSearchResult:
    query: str
    center: tuple[int, int]
    final_box: Box
    steps: list[VisualSearchStep]
    final_detection: FinalDetectionStep
    output_dir: str
    result_json: str
    elapsed_seconds: float


def run_llm_visual_search(
    image: Image.Image,
    *,
    query: str,
    client: VisionClient,
    output_root: Path,
    timestamp: str | None = None,
    rounds: int = 3,
    first_grid: tuple[int, int] = (3, 4),
    later_grid: tuple[int, int] = (3, 3),
    final_grid: tuple[int, int] = (5, 5),
    crop_padding: int = 40,
    final_crop_max_size: tuple[int, int] = (450, 350),
    judge: GridJudgeClient | None = None,
    max_judge_retries: int = 2,
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

        judge_attempts: list[GridJudgeAttempt] = []
        rejected_cell_ids: list[str] = []
        selected_local = _cell_by_id(local_cells, choice.cell_id)
        selected_box = _offset_box(selected_local.box, offset_x=current_box[0], offset_y=current_box[1])

        if judge is not None:
            for attempt_index in range(1, max_judge_retries + 2):
                selected_local = _cell_by_id(local_cells, choice.cell_id)
                selected_box = _offset_box(selected_local.box, offset_x=current_box[0], offset_y=current_box[1])
                judged_crop_box = expand_box(selected_box, padding=crop_padding, bounds=bounds)
                judged_crop = crop_box(image, judged_crop_box)
                judge_crop_path = output_dir / f"{round_index:02d}-judge-crop-attempt-{attempt_index}.png"
                judged_crop.save(judge_crop_path)

                judge_result = judge.judge_crop(query=query, image=judged_crop)
                judge_result_path = output_dir / f"{round_index:02d}-judge-result-attempt-{attempt_index}.json"
                attempt = GridJudgeAttempt(
                    attempt_index=attempt_index,
                    selected_cell_id=choice.cell_id,
                    selected_box=selected_box,
                    judged_crop_box=judged_crop_box,
                    crop_image=str(judge_crop_path),
                    contains_target=judge_result.contains_target,
                    confidence=judge_result.confidence,
                    rationale=judge_result.rationale,
                    visible_evidence=judge_result.visible_evidence,
                )
                judge_result_path.write_text(json.dumps(asdict(attempt), indent=2), encoding="utf-8")
                judge_attempts.append(attempt)

                if judge_result.contains_target:
                    break

                rejected_cell_ids.append(choice.cell_id)
                if attempt_index > max_judge_retries:
                    raise ValueError(
                        "Grid judge rejected all attempts "
                        f"for round {round_index}; rejected={rejected_cell_ids}; "
                        f"last_rationale={judge_result.rationale}"
                    )

                choice = client.revise_cell_choice(
                    query=query,
                    image=Image.open(grid_path).convert("RGB"),
                    cell_ids=[cell.id for cell in local_cells],
                    rejected_cell_ids=rejected_cell_ids,
                    judge_rationale=judge_result.rationale,
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
                judge_attempts=judge_attempts,
            )
        )
        current_box = expand_box(selected_box, padding=crop_padding, bounds=bounds)

    # Final step: iterative bbox refinement with validation
    final_crop = crop_box(image, current_box)
    final_crop_path = output_dir / "final-crop.png"
    final_crop.save(final_crop_path)

    detection = client.locate_icon_with_validation(query=query, image=final_crop, debug_dir=output_dir)
    if not detection.target_visible:
        raise ValueError(f"LLM reported target not visible in final crop: {detection.rationale}")

    icon_box = _offset_box(detection.icon_bbox, offset_x=current_box[0], offset_y=current_box[1])
    final_detection_path = output_dir / "final-detection.png"
    draw_box(final_crop, detection.icon_bbox, output_path=final_detection_path, label="icon_bbox")

    full_detection_path = output_dir / "full-detection.png"
    draw_box(image, icon_box, output_path=full_detection_path, label="icon_bbox")

    center = ((icon_box[0] + icon_box[2]) // 2, (icon_box[1] + icon_box[3]) // 2)
    final_detection = FinalDetectionStep(
        crop_box=current_box,
        icon_bbox_local=detection.icon_bbox,
        icon_bbox_screen=icon_box,
        confidence=detection.confidence,
        rationale=detection.rationale,
        crop_image=str(final_crop_path),
        detection_image=str(final_detection_path),
    )
    elapsed = time.perf_counter() - start_time
    print(f"[TIMING] Visual search completed in {elapsed:.2f} seconds")

    result_path = output_dir / "result.json"
    result = VisualSearchResult(
        query=query,
        center=center,
        final_box=icon_box,
        steps=steps,
        final_detection=final_detection,
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


def _union_boxes(boxes: list[Box]) -> Box:
    if not boxes:
        raise ValueError("Cannot compute union of empty box list")
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def _crop_is_small_enough(box: Box, *, max_size: tuple[int, int]) -> bool:
    return (box[2] - box[0]) <= max_size[0] and (box[3] - box[1]) <= max_size[1]
