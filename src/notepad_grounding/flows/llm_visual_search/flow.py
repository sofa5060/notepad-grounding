from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from notepad_grounding.shared.geometry import Box
from notepad_grounding.shared.geometry import build_grid_cells
from notepad_grounding.shared.geometry import expand_box
from notepad_grounding.shared.images import crop_box
from notepad_grounding.shared.images import draw_box
from notepad_grounding.shared.images import draw_grid_cells
from notepad_grounding.shared.llm import CellChoice
from notepad_grounding.shared.llm import IconDetection
from notepad_grounding.shared.llm import VisionClient


@dataclass(frozen=True)
class VisualSearchStep:
    round_index: int
    crop_box: Box
    selected_cell_id: str
    selected_box: Box
    confidence: float
    rationale: str
    grid_image: str


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
    crop_padding: int = 40,
    final_crop_max_size: tuple[int, int] = (450, 350),
) -> VisualSearchResult:
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
        selected_local = _cell_by_id(local_cells, choice)
        selected_box = _offset_box(selected_local.box, offset_x=current_box[0], offset_y=current_box[1])

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
            )
        )
        current_box = expand_box(selected_box, padding=crop_padding, bounds=bounds)

    final_crop = crop_box(image, current_box)
    final_crop_path = output_dir / "final-crop.png"
    final_crop.save(final_crop_path)
    detection = client.locate_icon(query=query, image=final_crop)
    if not detection.target_visible:
        raise ValueError(f"LLM reported target not visible in final crop: {detection.rationale}")

    icon_box = _offset_box(detection.icon_bbox, offset_x=current_box[0], offset_y=current_box[1])
    final_detection_path = output_dir / "final-detection.png"
    draw_box(final_crop, detection.icon_bbox, output_path=final_detection_path, label="icon_bbox")
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
    result_path = output_dir / "result.json"
    result = VisualSearchResult(
        query=query,
        center=center,
        final_box=icon_box,
        steps=steps,
        final_detection=final_detection,
        output_dir=str(output_dir),
        result_json=str(result_path),
    )
    result_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def _cell_by_id(cells, choice: CellChoice):
    for cell in cells:
        if cell.id == choice.cell_id:
            return cell
    raise ValueError(f"Unknown selected cell: {choice.cell_id}")


def _offset_box(box: Box, *, offset_x: int, offset_y: int) -> Box:
    return (
        box[0] + offset_x,
        box[1] + offset_y,
        box[2] + offset_x,
        box[3] + offset_y,
    )


def _crop_is_small_enough(box: Box, *, max_size: tuple[int, int]) -> bool:
    return (box[2] - box[0]) <= max_size[0] and (box[3] - box[1]) <= max_size[1]
