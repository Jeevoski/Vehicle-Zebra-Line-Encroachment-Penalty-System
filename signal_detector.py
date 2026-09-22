"""Traffic signal detection using HSV red-pixel density."""
from __future__ import annotations

import cv2
import numpy as np


class TrafficSignalDetector:
    """Classify a configured traffic-light ROI as RED or GREEN."""

    def __init__(
        self,
        signal_roi: list[int],
        lower_red_1: np.ndarray,
        upper_red_1: np.ndarray,
        lower_red_2: np.ndarray,
        upper_red_2: np.ndarray,
        red_pixel_ratio_threshold: float = 0.02,
    ) -> None:
        self.signal_roi = signal_roi
        self.lower_red_1 = lower_red_1
        self.upper_red_1 = upper_red_1
        self.lower_red_2 = lower_red_2
        self.upper_red_2 = upper_red_2
        self.red_pixel_ratio_threshold = red_pixel_ratio_threshold
        self.last_red_ratio = 0.0

    def detect(self, frame: np.ndarray) -> str:
        """Return RED when the ROI contains enough red pixels, else GREEN."""
        if frame is None or frame.size == 0:
            return "GREEN"

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self.signal_roi
        x1, x2 = sorted((max(0, x1), min(width, x2)))
        y1, y2 = sorted((max(0, y1), min(height, y2)))
        if x2 <= x1 or y2 <= y1:
            self.last_red_ratio = 0.0
            return "GREEN"

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask_1 = cv2.inRange(hsv, self.lower_red_1, self.upper_red_1)
        mask_2 = cv2.inRange(hsv, self.lower_red_2, self.upper_red_2)
        red_mask = cv2.bitwise_or(mask_1, mask_2)
        self.last_red_ratio = float(cv2.countNonZero(red_mask)) / red_mask.size
        return "RED" if self.last_red_ratio >= self.red_pixel_ratio_threshold else "GREEN"

    def annotate(self, frame: np.ndarray) -> None:
        """Draw the signal ROI and measured red ratio on a frame."""
        x1, y1, x2, y2 = self.signal_roi
        color = (0, 0, 255) if self.last_red_ratio >= self.red_pixel_ratio_threshold else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"Red ROI: {self.last_red_ratio:.1%}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
