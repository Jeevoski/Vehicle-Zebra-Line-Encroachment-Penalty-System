"""StripeGuard AI real-time traffic violation monitoring engine."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

import config
from anpr_engine import ANPREngine
from screen_capture import ScreenCapture
from signal_detector import TrafficSignalDetector
from stream_loader import ThreadedStreamLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("stripeguard")
DISPLAY_SIZE = (1280, 720)


def load_json_log(path: str) -> list[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def append_violation(entry: dict[str, Any]) -> None:
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


def crop_box(frame: np.ndarray, box: tuple[int, int, int, int] | list[int] | Any) -> np.ndarray:
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    height, width = frame.shape[:2]
    return frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)].copy()


def fit_preview(frame: np.ndarray) -> np.ndarray:
    """Fit a frame completely inside a fixed 16:9 canvas without distortion."""
    canvas_width, canvas_height = DISPLAY_SIZE
    height, width = frame.shape[:2]
    scale = min(canvas_width / width, canvas_height / height)
    resized = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=frame.dtype)
    x_offset = (canvas_width - resized.shape[1]) // 2
    y_offset = (canvas_height - resized.shape[0]) // 2
    canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
    return canvas


def scale_rois(frame: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Scale reference 1920x1080 ROIs to the actual inference frame."""
    frame_height, frame_width = frame.shape[:2]
    reference_width, reference_height = config.REFERENCE_FRAME_SIZE
    x_scale = float(frame_width) / float(reference_width)
    y_scale = float(frame_height) / float(reference_height)
    points = np.round(config.ZEBRA_ROI * [x_scale, y_scale]).astype(np.int32)
    signal_rois: list[list[int]] = []
    for x1, y1, x2, y2 in config.SIGNAL_ROIS:
        signal_rois.append([
            round(x1 * x_scale), round(y1 * y_scale),
            round(x2 * x_scale), round(y2 * y_scale),
        ])
    return points, signal_rois


def parse_screen_region(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    values = [int(item.strip()) for item in value.split(",")]
    if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
        raise ValueError("Screen region must be x,y,width,height with positive width and height")
    return (values[0], values[1], values[2], values[3])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor zebra-crossing incursions with StripeGuard AI")
    parser.add_argument("--source", default=config.VIDEO_SOURCE, help="Video path, camera index, or stream URL")
    parser.add_argument("--live", action="store_true", help="Use the configured YouTube live traffic stream")
    parser.add_argument("--screen", action="store_true", help="Capture the primary monitor instead of a video source")
    parser.add_argument("--screen-region", help="Capture x,y,width,height instead of the full primary monitor")
    parser.add_argument("--model", default=config.MODEL_PATH)
    parser.add_argument("--signal-mode", choices=("auto", "manual"), default="auto")
    parser.add_argument("--cpu-ocr", action="store_true", help="Force EasyOCR to use CPU")
    parser.add_argument("--headless", action="store_true", help="Process video without opening an OpenCV window")
    parser.add_argument("--force-red", action="store_true", help="Always treat signal as RED (ideal for game testing)")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold (default: 0.35)")
    parser.add_argument("--output-video", help="Save annotated detections to this video path")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_output_directories()
    model = YOLO(args.model)
    ocr = ANPREngine(gpu=config.OCR_GPU and not args.cpu_ocr, min_confidence=config.MIN_PLATE_CONFIDENCE)
    signal_detector = TrafficSignalDetector(
        config.SIGNAL_ROIS,
        config.LOWER_RED_1,
        config.UPPER_RED_1,
        config.LOWER_RED_2,
        config.UPPER_RED_2,
        config.LOWER_GREEN,
        config.UPPER_GREEN,
        config.RED_PIXEL_RATIO_THRESHOLD,
    )

    if args.screen and args.live:
        raise ValueError("Use either --screen or --live, not both")

    capture: Any = None
    if args.screen:
        capture = ScreenCapture(parse_screen_region(args.screen_region))
    else:
        source = config.LIVE_STREAM_URL if args.live else args.source
        is_youtube = str(source).startswith(("https://www.youtube.com/", "https://youtu.be/"))
        if is_youtube:
            capture = ThreadedStreamLoader(str(source), config.LIVE_STREAM_QUALITY)
        else:
            cam_source = int(source) if str(source).isdigit() else source
            capture = cv2.VideoCapture(cam_source)

    if capture is None or not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {args.source}")

    output_writer: cv2.VideoWriter | None = None
    if args.output_video:
        output_path = Path(args.output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
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

    manual_signal = "RED" if args.force_red else "GREEN"
    if args.force_red:
        args.signal_mode = "manual"
    previous_signal = manual_signal
    frame_count = 0
    frame_times: list[float] = []
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    active_zebra_roi: np.ndarray = np.asarray(config.ZEBRA_ROI).copy()
    rois_scaled = False
    ocr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anpr")
    pending_ocr: dict[int, tuple[Future[Any], dict[str, Any]]] = {}
    incursion_frames: dict[int, int] = {}

    try:
        if not args.headless:
            cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        while True:
            success, raw_frame = capture.read()
            if not success or raw_frame is None:
                break
            frame: np.ndarray = np.asarray(raw_frame)
            frame_count += 1
            if not rois_scaled:
                active_zebra_roi, active_signal_rois = scale_rois(frame)
                signal_detector.signal_rois = active_signal_rois
                rois_scaled = True
            start = time.perf_counter()
            completed_ids: list[int] = []
            for track_id, (future, entry) in list(pending_ocr.items()):
                if future.done():
                    try:
                        plate: Any = future.result()
                        entry["plate_number"] = str(getattr(plate, "text", plate))
                        entry["confidence"] = round(float(getattr(plate, "confidence", 0.0)), 4)
                        append_violation(entry)
                    except Exception:
                        LOGGER.exception("Error processing ANPR result for vehicle %s", track_id)
                    completed_ids.append(track_id)
            for track_id in completed_ids:
                pending_ocr.pop(track_id, None)
            detected_signal = signal_detector.detect(frame)
            signal = manual_signal if args.signal_mode == "manual" else detected_signal
            if signal == "RED" and previous_signal != "RED":
                penalized_ids.clear()
            previous_signal = signal

            results: Any = model.track(
                frame,
                persist=True,
                classes=config.VEHICLE_CLASSES,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )
            result = results[0]
            boxes = getattr(result, "boxes", None)
            tracked_ids: list[int] = []
            tracked_boxes: list[list[int]] = []
            if boxes is not None:
                box_id = getattr(boxes, "id", None)
                if box_id is not None:
                    tracked_ids = box_id.int().cpu().tolist()
                box_xyxy = getattr(boxes, "xyxy", None)
                if box_xyxy is not None:
                    tracked_boxes = box_xyxy.int().cpu().tolist()

            violating_ids: set[int] = set()
            for box, track_id in zip(tracked_boxes, tracked_ids):
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                if (x2 - x1) < 45 or (y2 - y1) < 35:
                    continue
                bottom_center = ((x1 + x2) // 2, y2)
                inside = (
                    cv2.pointPolygonTest(active_zebra_roi, (float(bottom_center[0]), float(bottom_center[1])), False) >= 0
                    or cv2.pointPolygonTest(active_zebra_roi, (float(x1), float(y2)), False) >= 0
                    or cv2.pointPolygonTest(active_zebra_roi, (float(x2), float(y2)), False) >= 0
                    or cv2.pointPolygonTest(active_zebra_roi, (float(bottom_center[0]), float(bottom_center[1])), True) >= -10.0
                )
                if inside:
                    if signal == "RED":
                        violating_ids.add(track_id)
                        if track_id not in penalized_ids:
                            penalized_ids.add(track_id)
                            vehicle_crop = crop_box(frame, (x1, y1, x2, y2))
                            timestamp = datetime.now(timezone.utc).astimezone()
                            stamp = timestamp.strftime("%Y%m%d_%H%M%S")
                            filename = f"ID{track_id}_PENDING_{stamp}.jpg"
                            output_path = Path(config.VIOLATIONS_DIR) / filename
                            cv2.imwrite(str(output_path), vehicle_crop)
                            entry: dict[str, Any] = {
                                "timestamp": timestamp.isoformat(),
                                "vehicle_id": track_id,
                                "plate_number": "PENDING",
                                "confidence": 0.0,
                                "file_path": str(output_path),
                            }
                            pending_ocr[track_id] = (ocr_executor.submit(ocr.extract_plate, vehicle_crop), entry)
                            violation_count += 1
                            LOGGER.info("Violation captured: %s", entry)
                            print(f"\n>>> [VIOLATION RECORDED!] Vehicle ID: {track_id} | Saved to: {output_path} <<<\n")
                            try:
                                import winsound
                                winsound.Beep(1200, 250)
                            except Exception:
                                pass
                    elif signal != "RED":
                        if frame_count % 20 == 0:
                            print(f"\n[INFO] Vehicle ID {track_id} is on zebra line, but signal is {signal} (legal)\n")

                is_violation = track_id in violating_ids
                color = (0, 0, 255) if is_violation else (0, 255, 0)
                thickness = 3 if is_violation else 2
                label = f"VIOLATION! ID: {track_id}" if is_violation else f"ID: {track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness, cv2.LINE_AA)
                if is_violation:
                    cv2.circle(frame, bottom_center, 7, (0, 255, 255), -1)

            roi_color = (0, 0, 255) if signal == "RED" else (0, 200, 0)
            cv2.polylines(frame, [active_zebra_roi], True, roi_color, 3)
            signal_detector.annotate(frame)
            elapsed = max(time.perf_counter() - start, 1e-6)
            frame_times.append(1.0 / elapsed)
            frame_times = frame_times[-20:]
            fps = sum(frame_times) / len(frame_times)
            cv2.rectangle(frame, (0, 0), (460, 72), (25, 25, 25), -1)
            cv2.putText(frame, f"SIGNAL: {signal} | VIOLATIONS: {violation_count}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, roi_color, 2, cv2.LINE_AA)
            measured_source_fps = float(getattr(capture, "fps", 0.0) or source_fps)
            cv2.putText(frame, f"Inference: {fps:.1f} | Source: {measured_source_fps:.1f} FPS", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
            if frame_count % 15 == 0:
                print(f"\r[StripeGuard] Frame: {frame_count:05d} | Signal: {signal:5s} | Vehicles Tracked: {len(tracked_ids)} | Violations: {violation_count} ", end="", flush=True)
            if output_writer is not None:
                output_writer.write(frame)
            key = -1
            if not args.headless:
                cv2.imshow(config.WINDOW_NAME, fit_preview(frame))
                key = cv2.waitKey(1) & 0xFF
            try:
                import msvcrt
                while msvcrt.kbhit():
                    ch = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                    if ch == "r":
                        manual_signal = "RED"
                        args.signal_mode = "manual"
                        print("\n[HOTKEY] Signal FORCED to RED\n")
                    elif ch == "g":
                        manual_signal = "GREEN"
                        args.signal_mode = "manual"
                        print("\n[HOTKEY] Signal FORCED to GREEN\n")
                    elif ch == "a":
                        args.signal_mode = "auto"
                        print("\n[HOTKEY] Switched back to AUTO\n")
                    elif ch == "q":
                        key = ord("q")
            except Exception:
                pass

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
        for future, entry in list(pending_ocr.values()):
            try:
                plate = future.result(timeout=2.0)
                entry["plate_number"] = str(getattr(plate, "text", plate))
                entry["confidence"] = round(float(getattr(plate, "confidence", 0.0)), 4)
                append_violation(entry)
            except Exception:
                LOGGER.exception("Unable to complete ANPR task for %s", entry.get("vehicle_id", "unknown"))
        ocr_executor.shutdown(wait=False, cancel_futures=True)
        if capture is not None:
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
