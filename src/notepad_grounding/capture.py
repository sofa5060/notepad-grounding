from __future__ import annotations
import mss
from PIL import Image


def capture_desktop(*, monitor_index: int = 1) -> Image.Image:
    with mss.mss() as screen_capture:
        monitors = screen_capture.monitors
        if monitor_index >= len(monitors):
            raise RuntimeError(f"Monitor {monitor_index} is unavailable; found {len(monitors) - 1}.")
        screenshot = screen_capture.grab(monitors[monitor_index])

    return Image.frombytes("RGB", screenshot.size, screenshot.rgb)
