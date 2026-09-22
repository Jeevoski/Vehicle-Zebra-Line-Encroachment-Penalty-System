"""Central configuration for StripeGuard AI."""
from pathlib import Path

import numpy as np

MODEL_PATH = "yolov8n.pt"
VIDEO_SOURCE = "data/traffic_sample.mp4"
VIOLATIONS_DIR = "violations"
LOG_JSON_PATH = "violations/violations.json"
LOG_CSV_PATH = "violations/violations.csv"

VEHICLE_CLASSES = [2, 3, 5, 7]
CONF_THRESHOLD = 0.4
RED_PIXEL_RATIO_THRESHOLD = 0.02
MIN_PLATE_CONFIDENCE = 0.25
OCR_GPU = False

# Coordinates are calibrated for the supplied 1920x1080 intersection footage.
# Replace them with values emitted by roi_selector.py for another camera angle.
ZEBRA_ROI = np.array(
    [[390, 345], [1900, 345], [1900, 535], [390, 535]],
    dtype=np.int32,
)
SIGNAL_ROI = [260, 135, 335, 260]

LOWER_RED_1 = np.array([0, 100, 100], dtype=np.uint8)
UPPER_RED_1 = np.array([10, 255, 255], dtype=np.uint8)
LOWER_RED_2 = np.array([160, 100, 100], dtype=np.uint8)
UPPER_RED_2 = np.array([179, 255, 255], dtype=np.uint8)

WINDOW_NAME = "StripeGuard AI"
CSV_FIELDS = [
    "timestamp",
    "vehicle_id",
    "plate_number",
    "confidence",
    "file_path",
]


def ensure_output_directories() -> None:
    """Create output directories and parent folders for configured logs."""
    Path(VIOLATIONS_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(LOG_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
