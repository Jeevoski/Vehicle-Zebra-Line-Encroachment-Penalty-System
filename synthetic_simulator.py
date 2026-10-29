"""Generate a lightweight synthetic zebra-crossing test video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 720
FPS = 20
ZEBRA_ROI = [[310, 505], [970, 505], [1040, 650], [240, 650]]
SIGNAL_ROI = [1080, 45, 1170, 190]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a lightweight synthetic traffic video")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--output", default="data/synthetic_traffic.mp4")
    parser.add_argument("--roi-output", default="data/synthetic_rois.json")
    return parser.parse_args()


def draw_vehicle(frame: np.ndarray, x: int, y: int, color: tuple[int, int, int], vehicle_id: int) -> None:
    cv2.rectangle(frame, (x, y), (x + 115, y + 65), color, -1)
    cv2.rectangle(frame, (x + 18, y - 25), (x + 92, y), color, -1)
    cv2.rectangle(frame, (x + 30, y - 19), (x + 80, y - 3), (145, 190, 210), -1)
    cv2.circle(frame, (x + 23, y + 65), 13, (25, 25, 25), -1)
    cv2.circle(frame, (x + 92, y + 65), 13, (25, 25, 25), -1)
    cv2.putText(frame, f"SIM-{vehicle_id}", (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)


def draw_frame(frame_index: int) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (58, 112, 78), dtype=np.uint8)
    cv2.rectangle(frame, (0, 365), (WIDTH, HEIGHT), (55, 55, 55), -1)
    cv2.line(frame, (0, 365), (WIDTH, 365), (210, 210, 210), 3)

    polygon = np.asarray(ZEBRA_ROI, dtype=np.int32)
    left, top = polygon[0]
    right, bottom = polygon[2]
    for index in range(9):
        x1 = left + index * (right - left) // 9
        x2 = left + (index + 1) * (right - left) // 9 - 8
        stripe = np.array([[x1, top], [x2, top], [x2 + 28, bottom], [x1 + 28, bottom]], dtype=np.int32)
        cv2.fillConvexPoly(frame, stripe, (235, 235, 235))

    signal = "RED" if (frame_index // (FPS * 8)) % 2 == 0 else "GREEN"
    light_x, light_y = 1125, 65
    cv2.rectangle(frame, (1080, 45), (1170, 190), (20, 20, 20), -1)
    for index, color in enumerate(((0, 0, 255), (0, 165, 255), (0, 200, 0))):
        active = (signal == "RED" and index == 0) or (signal == "GREEN" and index == 2)
        cv2.circle(frame, (light_x, light_y + index * 42), 16, color if active else (55, 55, 55), -1)
    cv2.putText(frame, signal, (1080, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    phase = frame_index / FPS
    vehicle_positions = [
        (int(180 + (phase * 52) % 950), 555, (65, 110, 210), 1),
        (int(820 - (phase * 34) % 680), 445, (210, 95, 55), 2),
        (int(360 + (phase * 42) % 650), 610, (75, 170, 105), 3),
    ]
    for x, y, color, vehicle_id in vehicle_positions:
        draw_vehicle(frame, x, y, color, vehicle_id)

    cv2.putText(frame, "SYNTHETIC TEST SCENE", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    roi_path = Path(args.roi_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    roi_path.write_text(json.dumps({"reference_frame_size": [WIDTH, HEIGHT], "zebra_roi": ZEBRA_ROI, "signal_rois": [SIGNAL_ROI]}, indent=2) + "\n", encoding="utf-8")

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video output: {output_path}")
    try:
        for frame_index in range(int(args.duration * FPS)):
            writer.write(draw_frame(frame_index))
    finally:
        writer.release()
    print(f"Recorded {output_path}")
    print(f"ROI metadata: {roi_path}")
    print("Copy the ROI values into config.py, then replay with:")
    print(f"python main.py --source {output_path} --signal-mode auto --headless --cpu-ocr --output-video violations/synthetic_annotated.mp4")


if __name__ == "__main__":
    main()
