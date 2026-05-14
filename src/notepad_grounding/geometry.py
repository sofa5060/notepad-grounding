from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class GridCell:
    id: str
    row: int
    col: int
    box: Box

    @property
    def center(self) -> tuple[int, int]:
        return ((self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2)


def build_grid_cells(
    box: Box,
    *,
    rows: int,
    cols: int,
    prefix: str = "A",
    id_fmt: str = "dash",
) -> list[GridCell]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    cells: list[GridCell] = []
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            cell_x1 = x1 + round(width * (col - 1) / cols)
            cell_y1 = y1 + round(height * (row - 1) / rows)
            cell_x2 = x1 + round(width * col / cols)
            cell_y2 = y1 + round(height * row / rows)
            if id_fmt == "rc":
                cell_id = f"R{row}C{col}"
            else:
                cell_id = f"{prefix}{row}-{col}"
            cells.append(
                GridCell(
                    id=cell_id,
                    row=row,
                    col=col,
                    box=(cell_x1, cell_y1, cell_x2, cell_y2),
                )
            )
    return cells


def cell_by_id(cells: Iterable[GridCell], cell_id: str) -> GridCell:
    for cell in cells:
        if cell.id == cell_id:
            return cell
    raise ValueError(f"Unknown cell: {cell_id}")


def offset_point(point: tuple[int, int], *, offset: tuple[int, int]) -> tuple[int, int]:
    return (point[0] + offset[0], point[1] + offset[1])


def clamp_box(box: Box, bounds: Box) -> Box:
    return (
        max(bounds[0], min(box[0], bounds[2])),
        max(bounds[1], min(box[1], bounds[3])),
        max(bounds[0], min(box[2], bounds[2])),
        max(bounds[1], min(box[3], bounds[3])),
    )


def expand_box(box: Box, *, padding: int, bounds: Box) -> Box:
    return clamp_box(
        (box[0] - padding, box[1] - padding, box[2] + padding, box[3] + padding),
        bounds,
    )
