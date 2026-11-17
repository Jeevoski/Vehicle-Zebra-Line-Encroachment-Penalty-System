"""Adaptive Night Vision and Adverse Weather (Rain, Fog, Haze) Enhancement Engine for StripeGuard AI."""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np


class AdaptiveVisionEnhancer:
    """Real-time adaptive contrast, gamma lift, and dehazing for night and adverse weather."""

    def __init__(self) -> None:
        self.clahe_night = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
        self.clahe_weather = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        self.clahe_plate = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6, 6))
        self._current_mode = "Clear Vision"
        # Precomputed gamma table for fast night vision lift (0.65 gamma)
        gamma = 0.65
        inv_gamma = 1.0 / gamma
        self._night_gamma_lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)

    def analyze_environment(self, frame: np.ndarray) -> dict[str, Any]:
        """Estimate luminance, dynamic range, and atmospheric conditions."""
        if frame is None or frame.size == 0:
            return {"luminance": 128.0, "contrast": 50.0, "is_night": False, "is_adverse_weather": False, "mode": "Clear Vision"}

        # Sample central road view for fast analysis
        h, w = frame.shape[:2]
        sample = frame[int(h * 0.25):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        std_dev = float(np.std(gray))

        is_night = mean_lum < 80.0
        # Low contrast, desaturated or hazy conditions indicate rain, overcast, or dense fog
        is_adverse_weather = (std_dev < 40.0 and mean_lum >= 80.0) or (std_dev < 32.0)

        mode = "Night Vision" if is_night else ("Adverse Weather (Rain/Fog)" if is_adverse_weather else "Clear Vision")
        self._current_mode = mode
        return {
            "luminance": round(mean_lum, 1),
            "contrast": round(std_dev, 1),
            "is_night": is_night,
            "is_adverse_weather": is_adverse_weather,
            "mode": mode,
        }

    def enhance_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str]:
        """Enhance frame adaptively based on detected night or rain/fog condition."""
        if frame is None or frame.size == 0:
            return frame, "Clear Vision"

        env = self.analyze_environment(frame)

        if env["is_night"]:
            # Night Vision Mode: Gamma expansion + LAB L-channel CLAHE
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_lifted = cv2.LUT(l, self._night_gamma_lut)
            l_enhanced = self.clahe_night.apply(l_lifted)
            enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)
            return enhanced, "Night Vision (Active)"

        if env["is_adverse_weather"]:
            # Rain & Fog Dehaze Mode: Saturation recovery + contrast stretch + unsharp edge restoration
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s_boost = cv2.multiply(s, 1.25)
            v_enhanced = self.clahe_weather.apply(v)
            dehazed = cv2.cvtColor(cv2.merge([h, s_boost, v_enhanced]), cv2.COLOR_HSV2BGR)
            gaussian = cv2.GaussianBlur(dehazed, (0, 0), sigmaX=1.6)
            crisp = cv2.addWeighted(dehazed, 1.35, gaussian, -0.35, 0)
            return crisp, "Adverse Weather Vision (Rain/Fog Dehaze)"

        return frame, "Clear Vision"

    def enhance_crop_for_anpr(self, crop: np.ndarray) -> np.ndarray:
        """Deep plate & vehicle enhancement handling rain glare and pitch darkness."""
        if crop is None or crop.size == 0:
            return crop
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe_plate.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
        return cv2.addWeighted(enhanced, 1.40, gaussian, -0.40, 0)

