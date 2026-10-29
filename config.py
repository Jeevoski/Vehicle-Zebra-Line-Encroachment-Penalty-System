"""Central configuration for StripeGuard AI."""
from pathlib import Path

import numpy as np

MODEL_PATH = "yolov8n.pt"
VIDEO_SOURCE = "data/traffic_sample.mp4"
LIVE_STREAM_URL = "https://www.youtube.com/live/sTF-6_xinUU"
LIVE_STREAM_QUALITY = "720p"
REFERENCE_FRAME_SIZE = (1920, 1080)
VIOLATIONS_DIR = "violations"
LOG_JSON_PATH = "violations/violations.json"
LOG_CSV_PATH = "violations/violations.csv"

VEHICLE_CLASSES = [2, 3, 5, 7]
CONF_THRESHOLD = 0.25
RED_PIXEL_RATIO_THRESHOLD = 0.04
MIN_PLATE_CONFIDENCE = 0.25
OCR_GPU = False

# Coordinates are calibrated for the Fresno stream's 1920x1080 foreground view.
# Replace them with values emitted by roi_selector.py when the camera angle changes.
ZEBRA_ROI = np.array(
    [[1113, 690], [1067, 736], [1559, 737], [1533, 688]],
    dtype=np.int32,
)
SIGNAL_ROIS = [[1209, 356, 1243, 433], [1418, 359, 1447, 435], [1612, 614, 1629, 668]]
SIGNAL_ROI = SIGNAL_ROIS[0]

LOWER_RED_1 = np.array([0, 50, 60], dtype=np.uint8)
UPPER_RED_1 = np.array([20, 255, 255], dtype=np.uint8)
LOWER_RED_2 = np.array([155, 50, 60], dtype=np.uint8)
UPPER_RED_2 = np.array([179, 255, 255], dtype=np.uint8)
LOWER_GREEN = np.array([35, 60, 60], dtype=np.uint8)
UPPER_GREEN = np.array([95, 255, 255], dtype=np.uint8)

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
