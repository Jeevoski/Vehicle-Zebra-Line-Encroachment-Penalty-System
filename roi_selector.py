"""Interactive ROI selection utility for StripeGuard AI."""
from __future__ import annotations

import cv2
import numpy as np

import config


class ROISelector:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.points: list[tuple[int, int]] = []
        self.mode = "zebra"

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
        elif self.mode == "signal" and len(self.points) == 2:
            cv2.rectangle(canvas, self.points[0], self.points[1], (255, 0, 0), 2)
        cv2.putText(canvas, f"Select {self.mode.upper()} ROI | ENTER: next/output | R: reset | Q: quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return canvas

    def _as_polygon(self):
        return np.array(self.points, dtype=np.int32)


def select_rois(frame):
    selector = ROISelector(frame)
    window = "StripeGuard ROI Selector"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, selector.callback)
    while True:
        cv2.imshow(window, selector.render())
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("ROI selection cancelled")
        if key == ord("r"):
            selector.points.clear()
        if key == 13:
            if selector.mode == "zebra" and len(selector.points) == 4:
                config.ZEBRA_ROI = np.array(selector.points, dtype=np.int32)
                selector.mode = "signal"
                selector.points.clear()
            elif selector.mode == "signal" and len(selector.points) == 2:
                (x1, y1), (x2, y2) = selector.points
                signal_roi = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                print("ZEBRA_ROI = np.array(%r, dtype=np.int32)" % config.ZEBRA_ROI.tolist())
                print("SIGNAL_ROI = %r" % signal_roi)
                cv2.destroyAllWindows()
                return
    

def main() -> None:
    capture = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {config.VIDEO_SOURCE}")
    success, frame = capture.read()
    capture.release()
    if not success:
        raise RuntimeError("Unable to read a frame for ROI selection")
    select_rois(frame)


if __name__ == "__main__":
    main()
