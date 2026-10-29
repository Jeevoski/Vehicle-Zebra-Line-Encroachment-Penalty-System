"""Traffic signal detection using position-aware bulb analysis and temporal smoothing."""
from __future__ import annotations

from collections import deque
from typing import Any
import cv2
import numpy as np


class TrafficSignalDetector:
    """Classify a configured traffic-light ROI as RED or GREEN."""

    def __init__(
        self,
        signal_roi: list[int] | list[list[int]] | Any,
        lower_red_1: np.ndarray,
        upper_red_1: np.ndarray,
        lower_red_2: np.ndarray,
        upper_red_2: np.ndarray,
        lower_green: np.ndarray,
        upper_green: np.ndarray,
        red_pixel_ratio_threshold: float = 0.03,
    ) -> None:
        rois: list[list[int]] = []
        if signal_roi:
            first = signal_roi[0]
            if isinstance(first, (list, tuple)):
                for item in signal_roi:
                    if isinstance(item, (list, tuple)):
                        rois.append([int(v) for v in item])
            elif isinstance(first, (int, float, np.integer)):
                rois.append([int(v) for v in signal_roi])
        self.signal_rois: list[list[int]] = rois
        self.red_pixel_ratio_threshold = 0.03
        self.last_red_ratio = 0.0
        self.last_green_ratio = 0.0
        self.last_state = "GREEN"
        self.roi_measurements: list[tuple[float, float]] = []
        self._history: deque[str] = deque(maxlen=5)

    def detect(self, frame: np.ndarray | None) -> str:
        """Return RED when the TOP bulb glows red, else GREEN when BOTTOM bulb glows green."""
        if frame is None or frame.size == 0:
            return self.last_state

        height, width = frame.shape[:2]
        measurements: list[tuple[float, float]] = []

        for roi in self.signal_rois:
            x1, y1, x2, y2 = roi
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                measurements.append((0.0, 0.0))
                continue

            crop = frame[y1:y2, x1:x2]
            crop_h, crop_w = crop.shape[:2]
            if crop_h < 6 or crop_w < 4:
                measurements.append((0.0, 0.0))
                continue

            # TOP 40%: Where the RED bulb physically is
            top_section = crop[0:int(crop_h * 0.40)]
            # BOTTOM 40%: Where the GREEN bulb physically is
            bot_section = crop[int(crop_h * 0.60):]

            # 1. Analyze RED in TOP section
            hsv_top = cv2.cvtColor(top_section, cv2.COLOR_BGR2HSV)
            red_mask = cv2.bitwise_or(
                cv2.inRange(hsv_top, np.array([0, 45, 60]), np.array([22, 255, 255])),
                cv2.inRange(hsv_top, np.array([150, 45, 60]), np.array([180, 255, 255])),
            )
            r_top = top_section[:, :, 2].astype(int)
            g_top = top_section[:, :, 1].astype(int)
            b_top = top_section[:, :, 0].astype(int)
            rgb_red = (r_top > 105) & (r_top > g_top * 1.25) & (r_top > b_top * 1.25)
            final_red = cv2.bitwise_or(red_mask, (rgb_red.astype(np.uint8) * 255))
            red_ratio = float(cv2.countNonZero(final_red)) / float(max(1, final_red.size))

            # 2. Analyze GREEN in BOTTOM section
            hsv_bot = cv2.cvtColor(bot_section, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv_bot, np.array([35, 45, 60]), np.array([95, 255, 255]))
            r_bot = bot_section[:, :, 2].astype(int)
            g_bot = bot_section[:, :, 1].astype(int)
            b_bot = bot_section[:, :, 0].astype(int)
            rgb_green = (g_bot > 95) & (g_bot > r_bot * 1.25) & (g_bot > b_bot * 0.9)
            final_green = cv2.bitwise_or(green_mask, (rgb_green.astype(np.uint8) * 255))
            green_ratio = float(cv2.countNonZero(final_green)) / float(max(1, final_green.size))

            measurements.append((red_ratio, green_ratio))

        self.roi_measurements = measurements
        best_red = max((r for r, _ in measurements), default=0.0)
        best_green = max((g for _, g in measurements), default=0.0)
        self.last_red_ratio = best_red
        self.last_green_ratio = best_green

        # Decision with hysteresis
        if best_red >= 0.030 and best_red > best_green:
            raw_state = "RED"
        elif best_green >= 0.030 and best_green > best_red:
            raw_state = "GREEN"
        else:
            raw_state = self.last_state

        # Temporal smoothing: 3 out of 5 frames needed to transition
        self._history.append(raw_state)
        red_votes = sum(1 for s in self._history if s == "RED")
        green_votes = sum(1 for s in self._history if s == "GREEN")

        if red_votes >= 3:
            self.last_state = "RED"
        elif green_votes >= 3:
            self.last_state = "GREEN"

        return self.last_state

    def annotate(self, frame: np.ndarray) -> None:
        """Draw the signal ROI and measured red ratio on a frame."""
        color = (0, 0, 255) if self.last_state == "RED" else (0, 200, 0)
        for index, roi in enumerate(self.signal_rois):
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            red_ratio, green_ratio = self.roi_measurements[index] if index < len(self.roi_measurements) else (0.0, 0.0)
            cv2.putText(frame, f"S{index + 1} R:{red_ratio:.0%} G:{green_ratio:.0%}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
