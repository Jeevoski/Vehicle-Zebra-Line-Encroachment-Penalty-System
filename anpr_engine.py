"""ANPR preprocessing and EasyOCR integration."""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import easyocr
import numpy as np


@dataclass(frozen=True)
class PlateResult:
    text: str
    confidence: float


class ANPREngine:
    """Extract a normalized alphanumeric plate string from a vehicle crop."""

    def __init__(self, gpu: bool = False, min_confidence: float = 0.25) -> None:
        self.min_confidence = min_confidence
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def extract_plate(self, vehicle_crop: np.ndarray) -> PlateResult:
        """Preprocess a crop and return its best OCR result."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return PlateResult("UNKNOWN", 0.0)

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        results = self.reader.readtext(filtered, detail=1, paragraph=False)
        if not results:
            return PlateResult("UNKNOWN", 0.0)

        candidates: list[PlateResult] = []
        for _, raw_text, confidence in results:
            cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
            score = float(confidence)
            if cleaned and score >= self.min_confidence:
                candidates.append(PlateResult(cleaned, score))
        if not candidates:
            return PlateResult("UNKNOWN", 0.0)
        return max(candidates, key=lambda item: item.confidence)
