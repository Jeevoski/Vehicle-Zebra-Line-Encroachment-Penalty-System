"""Interactive ROI selection utility for StripeGuard AI."""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np

import config
from screen_capture import ScreenCapture
from stream_loader import ThreadedStreamLoader


class ROISelector:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.points: list[tuple[int, int]] = []
        self.mode = "zebra"
        self.signal_rois: list[list[int]] = []

    def callback(self, event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        limit = 4 if self.mode == "zebra" else 2
        if len(self.points) < limit:
            self.points.append((x, y))

    def render(self):
        canvas = self.frame.copy()
        if self.mode == "zebra" and len(self.points) >= 2:
            for index, point in enumerate(self.points):
                cv2.circle(canvas, point, 5, (0, 255, 255), -1)
                cv2.putText(canvas, str(index + 1), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if len(self.points) == 4:
                cv2.polylines(canvas, [self._as_polygon()], True, (0, 255, 0), 2)
        elif self.mode == "signal":
            for x1, y1, x2, y2 in self.signal_rois:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if len(self.points) == 2:
                cv2.rectangle(canvas, self.points[0], self.points[1], (255, 0, 0), 2)
        instruction = "ENTER: save signal | S: finish & apply" if self.mode == "signal" else "ENTER: confirm zebra | R: reset | Q: quit"
        cv2.putText(canvas, f"Select {self.mode.upper()} ROI | {instruction}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return canvas

    def _as_polygon(self):
        return np.array(self.points, dtype=np.int32)


def save_to_config(frame_shape: tuple[int, ...], zebra_roi: np.ndarray, signal_rois: list[list[int]]) -> None:
    config_path = Path(__file__).resolve().parent / "config.py"
    if not config_path.exists():
        print(f"[WARN] {config_path} not found. Skipping auto-save.")
        return
    text = config_path.read_text(encoding="utf-8")
    h, w = frame_shape[:2]

    # Update REFERENCE_FRAME_SIZE
    text = re.sub(
        r"REFERENCE_FRAME_SIZE\s*=\s*\([^)]+\)",
        f"REFERENCE_FRAME_SIZE = ({w}, {h})",
        text,
    )
    # Update ZEBRA_ROI
    zebra_repr = f"ZEBRA_ROI = np.array(\n    {zebra_roi.tolist()},\n    dtype=np.int32,\n)"
    text = re.sub(
        r"ZEBRA_ROI\s*=\s*np\.array\([\s\S]*?dtype=np\.int32,?\s*\)",
        zebra_repr,
        text,
    )
    # Update SIGNAL_ROIS
    signal_repr = f"SIGNAL_ROIS = {signal_rois}"
    text = re.sub(
        r"SIGNAL_ROIS\s*=\s*\[[\s\S]*?\](?=\s*\nSIGNAL_ROI)",
        signal_repr,
        text,
    )
    config_path.write_text(text, encoding="utf-8")
    print("\n[SUCCESS] config.py automatically updated:")
    print(f"  - REFERENCE_FRAME_SIZE = ({w}, {h})")
    print(f"  - ZEBRA_ROI = {zebra_roi.tolist()}")
    print(f"  - SIGNAL_ROIS = {signal_rois}")


def select_rois(frame: np.ndarray, auto_save: bool = True) -> None:
    selector = ROISelector(frame)
    window = "StripeGuard ROI Selector"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, selector.callback)
    while True:
        cv2.imshow(window, selector.render())
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("ROI selection cancelled")
        if key == ord("r"):
            selector.points.clear()
            selector.mode = "zebra"
            selector.signal_rois.clear()
        if key == ord("s"):
            # Auto-commit any 2 uncommitted signal points before saving
            if selector.mode == "signal" and len(selector.points) == 2:
                (x1, y1), (x2, y2) = selector.points
                selector.signal_rois.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                selector.points.clear()
            if selector.mode == "signal" and selector.signal_rois:
                print("\n--- SELECTED ROI COORDINATES ---")
                print("ZEBRA_ROI = np.array(%r, dtype=np.int32)" % config.ZEBRA_ROI.tolist())
                print("SIGNAL_ROIS = %r" % selector.signal_rois)
                if auto_save:
                    save_to_config(frame.shape, config.ZEBRA_ROI, selector.signal_rois)
                cv2.destroyAllWindows()
                return
            else:
                print("[INFO] Please select at least one signal ROI before pressing 'S'.")
        if key == 13:
            if selector.mode == "zebra" and len(selector.points) == 4:
                config.ZEBRA_ROI = np.array(selector.points, dtype=np.int32)
                selector.mode = "signal"
                selector.points.clear()
            elif selector.mode == "signal" and len(selector.points) == 2:
                (x1, y1), (x2, y2) = selector.points
                selector.signal_rois.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                selector.points.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select StripeGuard signal and zebra ROIs")
    parser.add_argument("--screen", action="store_true", help="Select ROIs from the primary monitor (default)")
    parser.add_argument("--screen-region", help="Capture x,y,width,height for ROI selection")
    parser.add_argument("--source", help="Select ROIs from a local video file")
    parser.add_argument("--live", action="store_true", help="Select ROIs from the YouTube live stream")
    parser.add_argument("--delay", type=float, default=8.0, help="Seconds to switch to the game before capture")
    parser.add_argument("--no-save", action="store_true", help="Do not write directly to config.py")
    return parser.parse_args()


def parse_screen_region(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    values = [int(item.strip()) for item in value.split(",")]
    if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
        raise ValueError("Screen region must be x,y,width,height with positive width and height")
    return (values[0], values[1], values[2], values[3])


def main() -> None:
    args = parse_args()
    use_screen = args.screen or (not args.source and not args.live)

    if use_screen and args.delay > 0:
        for remaining in range(int(args.delay), 0, -1):
            print(f"Switch to ETS2 now... Capturing snapshot in {remaining}s")
            time.sleep(1.0)
        print("Capturing snapshot...")
        try:
            import winsound
            winsound.Beep(1000, 200)
        except Exception:
            pass

    if args.source:
        capture = cv2.VideoCapture(args.source)
    elif args.live:
        capture = ThreadedStreamLoader(config.LIVE_STREAM_URL, "1080p")
    else:
        capture = ScreenCapture(parse_screen_region(args.screen_region))

    success, frame = capture.read()
    capture.release()
    if not success or frame is None:
        source_name = "screen" if use_screen else (args.source or config.LIVE_STREAM_URL)
        raise RuntimeError(f"Unable to read a frame from {source_name}")
    select_rois(frame, auto_save=not args.no_save)


if __name__ == "__main__":
    main()
