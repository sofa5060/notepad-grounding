from __future__ import annotations

import platform

from PIL import Image


class DesktopCaptureError(RuntimeError):
    """Raised when the runtime cannot capture the desktop screenshot."""


def capture_desktop_screenshot(
    *,
    monitor_index: int = 1,
    require_windows: bool = True,
) -> Image.Image:
    """Capture the primary desktop monitor as an RGB Pillow image."""

    if require_windows and platform.system() != "Windows":
        raise DesktopCaptureError(
            "Desktop screenshot capture must be run inside the Windows VM. "
            "Use --allow-non-windows only for local smoke tests."
        )

    try:
        import mss
    except ImportError as exc:
        raise DesktopCaptureError(
            "Missing screenshot dependency 'mss'. Run 'uv sync' before retrying."
        ) from exc

    try:
        with mss.mss() as screen_capture:
            monitors = screen_capture.monitors
            if monitor_index >= len(monitors):
                raise DesktopCaptureError(
                    f"Monitor index {monitor_index} is unavailable; "
                    f"mss reported {len(monitors) - 1} monitor(s)."
                )

            screenshot = screen_capture.grab(monitors[monitor_index])
    except DesktopCaptureError:
        raise
    except Exception as exc:
        raise DesktopCaptureError(f"Unable to capture desktop screenshot: {exc}") from exc

    return Image.frombytes("RGB", screenshot.size, screenshot.rgb)

