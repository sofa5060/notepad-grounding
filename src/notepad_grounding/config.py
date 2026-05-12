from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


EXPECTED_SCREEN_SIZE = (1920, 1080)
DEFAULT_OUTPUT_DIR = Path("output/debug")


@dataclass(frozen=True)
class ScreenshotProofConfig:
    out_dir: Path = DEFAULT_OUTPUT_DIR
    expected_width: int = EXPECTED_SCREEN_SIZE[0]
    expected_height: int = EXPECTED_SCREEN_SIZE[1]
    strict_size: bool = False
    require_windows: bool = True

    @property
    def expected_size(self) -> tuple[int, int]:
        return (self.expected_width, self.expected_height)

