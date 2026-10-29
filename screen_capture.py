"""Windows desktop capture adapter for game and simulator input."""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np


class ScreenCapture:
    """Capture a monitor or screen region as OpenCV BGR frames."""

    def __init__(self, region: Optional[tuple[int, int, int, int]] = None, fps: float = 20.0) -> None:
        try:
            import mss
        except ImportError as exc:
            raise RuntimeError("Screen capture requires mss. Install it with: pip install mss") from exc

        self._sct = mss.mss()
        monitor = self._sct.monitors[1]
        if region is not None:
            left, top, width, height = region
            monitor = {"left": left, "top": top, "width": width, "height": height}
        self._monitor = monitor
        self._fps = fps
        self._frame_period = 1.0 / fps
        self._next_frame_time = 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        now = time.perf_counter()
        if now < self._next_frame_time:
            time.sleep(self._next_frame_time - now)
        self._next_frame_time = time.perf_counter() + self._frame_period
        screenshot = np.asarray(self._sct.grab(self._monitor))
        return True, cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._monitor["width"])
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._monitor["height"])
        if property_id == cv2.CAP_PROP_FPS:
            return self._fps
        return 0.0

    def release(self) -> None:
        self._sct.close()
