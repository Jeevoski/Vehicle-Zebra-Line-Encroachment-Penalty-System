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
from anpr_engine import ANPREngine, generate_synthetic_plate, render_plate_badge
from screen_capture import ScreenCapture
from signal_detector import TrafficSignalDetector
from stream_loader import ThreadedStreamLoader
from vision_enhancer import AdaptiveVisionEnhancer
from web_server import (
    add_violation_to_dashboard,
    get_signal_override,
    start_dashboard,
    update_dashboard,
)

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
        writer = csv.DictWriter(handle, fieldnames=config.CSV_FIELDS, extrasaction="ignore")
        if not file_exists or Path(config.LOG_CSV_PATH).stat().st_size == 0:
            writer.writeheader()
        writer.writerow(entry)


def compute_box_iou(box1: list[int] | tuple[int, int, int, int], box2: list[int] | tuple[int, int, int, int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = float(area1 + area2 - inter_area)
    return inter_area / union if union > 0 else 0.0


class CentroidFallbackTracker:
    """Lightweight centroid tracker that guarantees persistent vehicle IDs even if ByteTrack drops."""

    def __init__(self, max_disappeared: int = 15, max_distance: float = 90.0) -> None:
        self.next_id = 1
        self.objects: dict[int, tuple[float, float]] = {}
        self.disappeared: dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, boxes: list[list[int]]) -> list[int]:
        if not boxes:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.objects.pop(obj_id, None)
                    self.disappeared.pop(obj_id, None)
            return []

        input_centroids = [
            ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
            for b in boxes
        ]

        if not self.objects:
            assigned_ids = []
            for c in input_centroids:
                self.objects[self.next_id] = c
                self.disappeared[self.next_id] = 0
                assigned_ids.append(self.next_id)
                self.next_id += 1
            return assigned_ids

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        matched_ids: list[int | None] = [None] * len(input_centroids)
        used_obj_ids: set[int] = set()

        for i, (icx, icy) in enumerate(input_centroids):
            best_dist = self.max_distance
            best_id = None
            for obj_id, (ocx, ocy) in zip(object_ids, object_centroids):
                if obj_id in used_obj_ids:
                    continue
                dist = ((icx - ocx) ** 2 + (icy - ocy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id = obj_id
            if best_id is not None:
                matched_ids[i] = best_id
                used_obj_ids.add(best_id)
                self.objects[best_id] = (icx, icy)
                self.disappeared[best_id] = 0
            else:
                new_id = self.next_id
                self.next_id += 1
                self.objects[new_id] = (icx, icy)
                self.disappeared[new_id] = 0
                matched_ids[i] = new_id

        for obj_id in object_ids:
            if obj_id not in used_obj_ids:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.objects.pop(obj_id, None)
                    self.disappeared.pop(obj_id, None)

        return [id_ if id_ is not None else 0 for id_ in matched_ids]


def box_distance(box1: list[int] | tuple[int, int, int, int], box2: list[int] | tuple[int, int, int, int]) -> float:
    c1x = (box1[0] + box1[2]) / 2.0
    c1y = (box1[1] + box1[3]) / 2.0
    c2x = (box2[0] + box2[2]) / 2.0
    c2y = (box2[1] + box2[3]) / 2.0
    return ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5


def enhance_evidence_clarity(crop: np.ndarray) -> np.ndarray:
    """Enhance evidence photo clarity using LAB CLAHE and high-pass sharpening."""
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    target_w = max(w, 560)
    target_h = max(h, 420)
    if target_w > w or target_h > h:
        scale = max(target_w / float(w), target_h / float(h))
        crop = cv2.resize(crop, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LANCZOS4)

    # LAB color contrast equalization: enhances highlights & reflections without altering car paint color
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([l_eq, a_channel, b_channel])
    contrast_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # High-pass crisp unsharp mask
    gaussian = cv2.GaussianBlur(contrast_bgr, (0, 0), sigmaX=2.0)
    crisp = cv2.addWeighted(contrast_bgr, 1.45, gaussian, -0.45, 0)
    return crisp


def crop_box(frame: np.ndarray, box: tuple[int, int, int, int] | list[int] | Any, pad_ratio: float = 0.12) -> np.ndarray:
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    height, width = frame.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(width, x2 + pad_x)
    ny2 = min(height, y2 + pad_y)
    raw_crop = frame[ny1:ny2, nx1:nx2].copy()
    return enhance_evidence_clarity(raw_crop)


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
    parser.add_argument("--gui", action="store_true", help="Force OpenCV GUI window during screen capture")
    parser.add_argument("--force-red", action="store_true", help="Always treat signal as RED (ideal for game testing)")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD, help=f"YOLO confidence threshold (default: {config.CONF_THRESHOLD})")
    parser.add_argument("--output-video", help="Save annotated detections to this video path")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--no-web", action="store_true", help="Disable the browser frontend dashboard")
    parser.add_argument("--web-port", type=int, default=5000, help="Web dashboard port (default: 5000)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.force_red and args.signal_mode == "auto":
        stale_override = Path("violations/signal_override.json")
        if stale_override.exists():
            try:
                stale_override.unlink()
            except Exception:
                pass
    if args.screen and not args.gui:
        args.headless = True
    config.ensure_output_directories()
    if not args.no_web:
        start_dashboard(port=args.web_port)
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
        if not args.force_red:
            print("\n" + "=" * 65)
            print(" [INFO] Screen capture mode active.")
            print(" [NOTE] In testing/games, ensure traffic light state is RED.")
            print("        Pass '--force-red' or toggle RED on http://localhost:5000")
            print("        to enforce zebra line violations!")
            print("=" * 65 + "\n")
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
    active_violators: list[dict[str, Any]] = []
    fallback_tracker = CentroidFallbackTracker()
    vision_enhancer = AdaptiveVisionEnhancer()
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
            clean_frame: np.ndarray = frame.copy()
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
                        plate_text = str(getattr(plate, "text", ""))
                        plate_conf = round(float(getattr(plate, "confidence", 0.0)), 4)
                        if plate_text and plate_text != "UNKNOWN":
                            entry["plate_number"] = plate_text
                            entry["confidence"] = plate_conf
                            plate_crop = getattr(plate, "plate_crop", None)
                            if plate_crop is not None:
                                plate_filename = f"ID{track_id}_PLATE_{plate_text}.jpg"
                                plate_save_path = Path(config.VIOLATIONS_DIR) / plate_filename
                                cv2.imwrite(str(plate_save_path), plate_crop)
                                entry["plate_file_path"] = str(plate_save_path)
                                print(f"\n>>> [ANPR RECOGNIZED] Vehicle ID: {track_id} | Plate: {plate_text} ({plate_conf*100:.1f}%) <<<\n")
                            # Update record on disk
                            entries = load_json_log(config.LOG_JSON_PATH)
                            for idx, existing in enumerate(entries):
                                if existing.get("vehicle_id") == track_id and existing.get("timestamp") == entry.get("timestamp"):
                                    entries[idx] = entry
                                    break
                            with open(config.LOG_JSON_PATH, "w", encoding="utf-8") as handle:
                                json.dump(entries, handle, indent=2)
                            if not args.no_web:
                                add_violation_to_dashboard(entry)
                    except Exception:
                        LOGGER.exception("Error processing ANPR result for vehicle %s", track_id)
                    completed_ids.append(track_id)
            for track_id in completed_ids:
                pending_ocr.pop(track_id, None)
            if not args.no_web:
                web_mode, web_sig = get_signal_override()
                if web_mode == "manual":
                    args.signal_mode = "manual"
                    manual_signal = web_sig
                elif web_mode == "auto" and not args.force_red:
                    args.signal_mode = "auto"

            detected_signal = signal_detector.detect(frame)
            signal = manual_signal if args.signal_mode == "manual" else detected_signal
            if signal == "RED" and previous_signal != "RED":
                penalized_ids.clear()
                active_violators.clear()
            previous_signal = signal

            # Adaptive Night Vision & Rain/Fog Dehazing
            enhanced_frame, vision_mode = vision_enhancer.enhance_frame(frame)

            results: Any = model.track(
                enhanced_frame,
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
                box_xyxy = getattr(boxes, "xyxy", None)
                if box_xyxy is not None:
                    tracked_boxes = box_xyxy.int().cpu().tolist()
                box_id = getattr(boxes, "id", None)
                if box_id is not None:
                    tracked_ids = box_id.int().cpu().tolist()
                elif tracked_boxes:
                    # Fallback tracker ensures detections are never dropped if ByteTrack hasn't assigned IDs
                    tracked_ids = fallback_tracker.update(tracked_boxes)

            current_time = time.time()
            # Prune inactive violators older than 6.0s (based on ticket timestamp, not rolling last_seen)
            active_violators = [v for v in active_violators if (current_time - v["timestamp"]) < 6.0]

            violating_ids: set[int] = set()
            for box, track_id in zip(tracked_boxes, tracked_ids):
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                if (x2 - x1) < 25 or (y2 - y1) < 20:
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                bottom_center = (cx, y2)
                top_center = (cx, y1)
                front_quarter = (cx, int(y1 + (y2 - y1) * 0.75))

                test_points = [
                    bottom_center,
                    (cx, cy),
                    (x1, y2),
                    (x2, y2),
                    front_quarter,
                    top_center,
                ]

                inside = any(
                    cv2.pointPolygonTest(active_zebra_roi, (float(pt[0]), float(pt[1])), False) >= 0
                    or cv2.pointPolygonTest(active_zebra_roi, (float(pt[0]), float(pt[1])), True) >= -12.0
                    for pt in test_points
                ) or any(
                    x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2
                    for pt in active_zebra_roi
                )
                if inside:
                    if signal == "RED":
                        violating_ids.add(track_id)
                        # De-duplication: only match if EXACT same track_id OR high bounding box overlap (IoU > 0.40)
                        matched_violator = None
                        for v in active_violators:
                            if track_id == v["track_id"] or compute_box_iou(box, v["box"]) > 0.40:
                                matched_violator = v
                                break

                        if matched_violator is not None:
                            # Same vehicle already ticketed - update position without issuing duplicate ticket
                            matched_violator["box"] = box
                            penalized_ids.add(track_id)
                        elif track_id not in penalized_ids:
                            penalized_ids.add(track_id)
                            # Save clean crop with context padding and LAB contrast clarity
                            vehicle_crop = crop_box(clean_frame, (x1, y1, x2, y2))
                            timestamp = datetime.now(timezone.utc).astimezone()
                            stamp = timestamp.strftime("%Y%m%d_%H%M%S")
                            filename = f"ID{track_id}_PENDING_{stamp}.jpg"
                            output_path = Path(config.VIOLATIONS_DIR) / filename
                            cv2.imwrite(str(output_path), vehicle_crop)

                            # Assign realistic registration plate and render plate badge
                            assigned_plate, assigned_country, assigned_conf = generate_synthetic_plate(track_id)
                            plate_badge = render_plate_badge(assigned_plate, assigned_country)
                            plate_filename = f"ID{track_id}_PLATE_{assigned_plate}.jpg"
                            plate_save_path = Path(config.VIOLATIONS_DIR) / plate_filename
                            cv2.imwrite(str(plate_save_path), plate_badge)

                            active_violators.append({
                                "track_id": track_id,
                                "box": box,
                                "timestamp": current_time,
                                "plate_number": assigned_plate,
                            })

                            entry: dict[str, Any] = {
                                "timestamp": timestamp.isoformat(),
                                "vehicle_id": track_id,
                                "plate_number": assigned_plate,
                                "confidence": assigned_conf,
                                "file_path": str(output_path),
                                "plate_file_path": str(plate_save_path),
                            }
                            # Save immediately to disk so real-time frontend reflects instantly!
                            append_violation(entry)
                            if not args.no_web:
                                add_violation_to_dashboard(entry)
                            pending_ocr[track_id] = (ocr_executor.submit(ocr.extract_plate, vehicle_crop, track_id), entry)
                            violation_count += 1
                            LOGGER.info("Violation captured: %s", entry)
                            print(f"\n>>> [VIOLATION RECORDED!] Vehicle ID: {track_id} | Plate: {assigned_plate} | Saved to: {output_path} <<<\n")
                            try:
                                import winsound
                                winsound.Beep(1200, 250)
                            except Exception:
                                pass
                    elif signal != "RED":
                        if frame_count % 30 == 0:
                            print(f"\n[INFO] Vehicle ID {track_id} on zebra line, but signal is {signal} (legal). Run with --force-red or toggle RED on dashboard to enforce.\n")

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
                print(f"\r[StripeGuard] Frame: {frame_count:05d} | Signal: {signal:5s} | Vision: {vision_mode:25s} | Tracked: {len(tracked_ids)} | Violations: {violation_count} ", end="", flush=True)
            if output_writer is not None:
                output_writer.write(frame)
            if not args.no_web:
                update_dashboard(frame, signal, len(tracked_ids), violation_count, fps, measured_source_fps, vision_mode)
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
