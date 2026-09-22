"""StripeGuard AI real-time traffic violation monitoring engine."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import config
from anpr_engine import ANPREngine
from signal_detector import TrafficSignalDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("stripeguard")


def load_json_log(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def append_violation(entry: dict) -> None:
    config.ensure_output_directories()
    entries = load_json_log(config.LOG_JSON_PATH)
    entries.append(entry)
    with open(config.LOG_JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)

    file_exists = Path(config.LOG_CSV_PATH).exists()
    with open(config.LOG_CSV_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=config.CSV_FIELDS)
        if not file_exists or Path(config.LOG_CSV_PATH).stat().st_size == 0:
            writer.writeheader()
        writer.writerow(entry)


def crop_box(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    return frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor zebra-crossing incursions with StripeGuard AI")
    parser.add_argument("--source", default=config.VIDEO_SOURCE, help="Video path, camera index, or stream URL")
    parser.add_argument("--model", default=config.MODEL_PATH)
    parser.add_argument("--signal-mode", choices=("auto", "manual"), default="auto")
    parser.add_argument("--cpu-ocr", action="store_true", help="Force EasyOCR to use CPU")
    parser.add_argument("--headless", action="store_true", help="Process video without opening an OpenCV window")
    parser.add_argument("--output-video", help="Save annotated detections to this video path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_output_directories()
    model = YOLO(args.model)
    ocr = ANPREngine(gpu=config.OCR_GPU and not args.cpu_ocr, min_confidence=config.MIN_PLATE_CONFIDENCE)
    signal_detector = TrafficSignalDetector(
        config.SIGNAL_ROI,
        config.LOWER_RED_1,
        config.UPPER_RED_1,
        config.LOWER_RED_2,
        config.UPPER_RED_2,
        config.RED_PIXEL_RATIO_THRESHOLD,
    )

    source = int(args.source) if str(args.source).isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    output_writer = None
    if args.output_video:
        output_path = Path(args.output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        output_writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not output_writer.isOpened():
            raise RuntimeError(f"Unable to open annotated video output: {output_path}")

    penalized_ids: set[int] = set()
    violation_count = len(load_json_log(config.LOG_JSON_PATH))
    manual_signal = "GREEN"
    previous_signal = "GREEN"
    frame_times: list[float] = []

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            start = time.perf_counter()
            detected_signal = signal_detector.detect(frame)
            signal = manual_signal if args.signal_mode == "manual" else detected_signal
            if signal == "RED" and previous_signal != "RED":
                penalized_ids.clear()
            previous_signal = signal

            results = model.track(
                frame,
                persist=True,
                classes=config.VEHICLE_CLASSES,
                conf=config.CONF_THRESHOLD,
                verbose=False,
            )
            result = results[0]
            boxes = result.boxes
            tracked_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
            tracked_boxes = boxes.xyxy.int().cpu().tolist()
            violating_ids: set[int] = set()
            for box, track_id in zip(tracked_boxes, tracked_ids):
                x1, y1, x2, y2 = box
                bottom_center = (int((x1 + x2) / 2), int(y2))
                inside = cv2.pointPolygonTest(config.ZEBRA_ROI, bottom_center, False) >= 0
                if inside and signal == "RED":
                    violating_ids.add(track_id)
                    if track_id not in penalized_ids:
                        penalized_ids.add(track_id)
                        vehicle_crop = crop_box(frame, (x1, y1, x2, y2))
                        plate = ocr.extract_plate(vehicle_crop)
                        timestamp = datetime.now(timezone.utc).astimezone()
                        stamp = timestamp.strftime("%Y%m%d_%H%M%S")
                        filename = f"ID{track_id}_{plate.text}_{stamp}.jpg"
                        output_path = Path(config.VIOLATIONS_DIR) / filename
                        cv2.imwrite(str(output_path), vehicle_crop)
                        entry = {
                            "timestamp": timestamp.isoformat(),
                            "vehicle_id": track_id,
                            "plate_number": plate.text,
                            "confidence": round(plate.confidence, 4),
                            "file_path": str(output_path),
                        }
                        append_violation(entry)
                        violation_count += 1
                        LOGGER.info("Violation recorded: %s", entry)

                color = (0, 0, 255) if track_id in violating_ids else (255, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {track_id}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            roi_color = (0, 0, 255) if signal == "RED" else (0, 200, 0)
            cv2.polylines(frame, [config.ZEBRA_ROI], True, roi_color, 3)
            signal_detector.annotate(frame)
            elapsed = max(time.perf_counter() - start, 1e-6)
            frame_times.append(1.0 / elapsed)
            frame_times = frame_times[-20:]
            fps = sum(frame_times) / len(frame_times)
            cv2.rectangle(frame, (0, 0), (460, 72), (25, 25, 25), -1)
            cv2.putText(frame, f"SIGNAL: {signal} | VIOLATIONS: {violation_count}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, roi_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.1f} | Mode: {args.signal_mode}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
            if output_writer is not None:
                output_writer.write(frame)
            key = -1
            if not args.headless:
                cv2.imshow(config.WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                manual_signal = "RED"
                args.signal_mode = "manual"
            elif key == ord("g"):
                manual_signal = "GREEN"
                args.signal_mode = "manual"
            elif key == ord("a"):
                args.signal_mode = "auto"
    finally:
        capture.release()
        if output_writer is not None:
            output_writer.release()
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                LOGGER.warning("OpenCV GUI is unavailable; video processing completed headlessly")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info("Shutdown requested")
    except Exception:
        LOGGER.exception("StripeGuard AI stopped unexpectedly")
        raise
