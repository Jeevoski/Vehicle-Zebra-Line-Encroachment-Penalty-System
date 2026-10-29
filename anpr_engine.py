"""High-performance ANPR preprocessing and EasyOCR integration for StripeGuard AI."""
from __future__ import annotations

import os
# Prevent OpenMP runtime library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import re
from dataclasses import dataclass

import cv2
import easyocr
import numpy as np

LOGGER = logging.getLogger("stripeguard.anpr")

# Blacklist words commonly misread from vehicle grilles, decals, or UI text
BLACKLIST_WORDS = {
    "VIOLATION", "VIOLATIQ", "PENDING", "UNKNOWN", "STRIPEGUARD",
    "TRACK", "STOP", "RED", "GREEN", "YELLOW", "TRAFFIC", "SIGNAL"
}


@dataclass(frozen=True)
class PlateResult:
    text: str
    confidence: float
    plate_crop: np.ndarray | None = None


class ANPREngine:
    """Multi-stage license plate localization, enhancement, and OCR extraction."""

    def __init__(self, gpu: bool = False, min_confidence: float = 0.15) -> None:
        self.min_confidence = min_confidence
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def enhance_crop(self, crop: np.ndarray) -> np.ndarray:
        """Upscale, sharpen, and contrast-equalize a candidate plate region."""
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return crop

        # Super-resolution / Upscaling: target minimum plate height of 120px
        target_h = max(120, h)
        scale = target_h / float(h)
        target_w = max(int(w * scale), 200)
        upscaled = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Grayscale conversion
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY) if len(upscaled.shape) == 3 else upscaled

        # Unsharp masking (crispen character strokes)
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.5)
        sharpened = cv2.addWeighted(gray, 2.2, gaussian, -1.2, 0)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        contrast = clahe.apply(sharpened)

        # Bilateral filter (smooth noise while preserving crisp character edges)
        filtered = cv2.bilateralFilter(contrast, 7, 50, 50)
        return filtered

    def get_plate_candidates(self, vehicle_crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Extract high-probability bounding boxes for license plate locations."""
        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 20:
            return [("full", vehicle_crop)]

        candidates: list[tuple[str, np.ndarray]] = []

        # 1. Morphological Gradient / Contour Search for rectangular plate aspect ratios
        try:
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
            sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
            _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            vehicle_area = w * h
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = float(cw) / max(ch, 1)
                area = cw * ch
                # Plate typical aspect ratio: 2.0 - 5.8, area: 0.5% - 15% of vehicle
                if 2.0 <= aspect <= 5.8 and (0.004 * vehicle_area) <= area <= (0.16 * vehicle_area):
                    px = max(0, x - int(cw * 0.12))
                    py = max(0, y - int(ch * 0.15))
                    pw = min(w - px, int(cw * 1.24))
                    ph = min(h - py, int(ch * 1.30))
                    c_crop = vehicle_crop[py:py + ph, px:px + pw]
                    if c_crop.size > 0:
                        candidates.append(("contour", c_crop))
        except Exception:
            LOGGER.debug("Morphological plate contour detection skipped")

        # 2. Geometric Quadrant Slicing (Guarantees coverage across various camera angles)
        # Lower Center Bumper (Standard head-on view)
        center_bumper = vehicle_crop[int(h * 0.50):int(h * 0.98), int(w * 0.15):int(w * 0.85)]
        if center_bumper.size > 0:
            candidates.append(("lower_center", center_bumper))

        # Lower Left Bumper (Left-angled or turning vehicle)
        left_bumper = vehicle_crop[int(h * 0.48):int(h * 0.98), 0:int(w * 0.55)]
        if left_bumper.size > 0:
            candidates.append(("lower_left", left_bumper))

        # Lower Right Bumper (Right-angled vehicle)
        right_bumper = vehicle_crop[int(h * 0.48):int(h * 0.98), int(w * 0.45):w]
        if right_bumper.size > 0:
            candidates.append(("lower_right", right_bumper))

        # Full vehicle crop fallback
        candidates.append(("full_vehicle", vehicle_crop))
        return candidates

    def extract_plate(self, vehicle_crop: np.ndarray) -> PlateResult:
        """Preprocess crop, localize plate candidates, and return best OCR string."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return PlateResult("UNKNOWN", 0.0)

        candidates = self.get_plate_candidates(vehicle_crop)
        best_result = PlateResult("UNKNOWN", 0.0)
        best_score: float = -1.0

        for candidate_name, raw_crop in candidates:
            enhanced = self.enhance_crop(raw_crop)
            try:
                # Use strict alphanumeric allowlist
                results = self.reader.readtext(
                    enhanced,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    detail=1,
                    paragraph=False,
                )
            except Exception:
                continue

            for _, raw_text, confidence in results:
                cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
                conf = float(confidence)

                # Skip blacklist noise words and short fragments
                if cleaned in BLACKLIST_WORDS or len(cleaned) < 3:
                    continue
                if conf < self.min_confidence:
                    continue

                # Heuristic plate scoring
                score = conf
                has_digits = bool(re.search(r"\d", cleaned))
                has_letters = bool(re.search(r"[A-Z]", cleaned))

                # Real license plates typically combine digits and letters
                if has_digits and has_letters:
                    score += 0.35
                if 4 <= len(cleaned) <= 10:
                    score += 0.20
                if candidate_name != "full_vehicle":
                    score += 0.10

                if score > best_score:
                    best_score = score
                    best_result = PlateResult(
                        text=cleaned,
                        confidence=round(conf, 4),
                        plate_crop=enhanced,
                    )

        return best_result
