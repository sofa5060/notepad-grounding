from __future__ import annotations

from dataclasses import dataclass

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
) -> list[GridCell]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    cells: list[GridCell] = []
    for row in range(rows):
        for col in range(cols):
            cell_x1 = x1 + round(width * col / cols)
            cell_y1 = y1 + round(height * row / rows)
            cell_x2 = x1 + round(width * (col + 1) / cols)
            cell_y2 = y1 + round(height * (row + 1) / rows)
            cells.append(
                GridCell(
                    id=f"{prefix}{row + 1}-{col + 1}",
                    row=row,
                    col=col,
                    box=(cell_x1, cell_y1, cell_x2, cell_y2),
                )
            )
    return cells


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
