from __future__ import annotations

import platform

from PIL import Image


class CaptureError(RuntimeError):
    """Raised when screenshot capture cannot run."""


def capture_desktop(*, require_windows: bool = True, monitor_index: int = 1) -> Image.Image:
    if require_windows and platform.system() != "Windows":
        raise CaptureError("Live desktop capture must run inside Windows.")

    try:
        import mss
    except ImportError as exc:
        raise CaptureError("Missing screenshot dependency 'mss'. Run 'uv sync'.") from exc

    try:
        with mss.mss() as screen_capture:
            monitors = screen_capture.monitors
            if monitor_index >= len(monitors):
                raise CaptureError(f"Monitor {monitor_index} is unavailable; found {len(monitors) - 1}.")
            screenshot = screen_capture.grab(monitors[monitor_index])
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"Unable to capture desktop screenshot: {exc}") from exc

    return Image.frombytes("RGB", screenshot.size, screenshot.rgb)
