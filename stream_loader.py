"""Low-latency YouTube HLS ingestion for StripeGuard AI."""
from __future__ import annotations

import logging
import re
import threading
import time

import cv2

LOGGER = logging.getLogger("stripeguard.stream")
DEFAULT_LIVE_STREAM_URL = "https://www.youtube.com/live/sTF-6_xinUU"
PREFERRED_QUALITIES = ("720p", "1080p", "best")


def _select_stream(streams: dict, preferred_quality: str = "720p"):
    """Select a preferred quality, falling back to the highest available rendition."""
    for quality in (preferred_quality, "720p", "1080p", "best"):
        if streams.get(quality) is not None:
            return streams[quality]
    candidates = []
    for name, stream in streams.items():
        match = re.search(r"(\d{3,4})p", str(name).lower())
        if match:
            candidates.append((int(match.group(1)), stream))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    for quality in PREFERRED_QUALITIES:
        stream = streams.get(quality)
        if stream is not None:
            return stream
    raise RuntimeError("Streamlink found no playable video streams")


def get_youtube_stream(url: str = DEFAULT_LIVE_STREAM_URL, quality: str = "720p") -> cv2.VideoCapture:
    """Resolve a YouTube URL with Streamlink and return an opened VideoCapture."""
    if not url.startswith(("https://www.youtube.com/", "https://youtu.be/")):
        raise ValueError(f"Expected a YouTube URL, received: {url}")

    try:
        from streamlink import streams
    except ImportError as exc:
        raise RuntimeError(
            "Streamlink is required for YouTube input. Install it with: pip install streamlink"
        ) from exc

    try:
        available = streams(url)
        selected = _select_stream(available, quality)
        stream_url = selected.to_url()
    except Exception as exc:
        raise RuntimeError(f"Unable to resolve YouTube stream: {url}") from exc

    capture = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not open the resolved YouTube stream")
    LOGGER.info("Opened YouTube stream at %s", getattr(selected, "__class__", type(selected)).__name__)
    return capture


class ThreadedStreamLoader:
    """Continuously decode a live stream while retaining only the latest frame."""

    def __init__(self, url: str = DEFAULT_LIVE_STREAM_URL, quality: str = "720p") -> None:
        self._url = url
        self._quality = quality
        self._capture = get_youtube_stream(url, quality)
        self._condition = threading.Condition()
        self._frame = None
        self._stopped = False
        self._decoded_frames = 0
        self._fps = 0.0
        self._fps_started = time.perf_counter()
        self._reader = threading.Thread(target=self._read_loop, name="stream-reader", daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        while not self._stopped:
            try:
                success, frame = self._capture.read()
            except cv2.error as exc:
                LOGGER.warning("Live stream decode interrupted: %s", exc)
                success, frame = False, None
            if success:
                self._decoded_frames += 1
                elapsed = time.perf_counter() - self._fps_started
                if elapsed >= 1.0:
                    self._fps = self._decoded_frames / elapsed
                    self._decoded_frames = 0
                    self._fps_started = time.perf_counter()
                with self._condition:
                    self._frame = frame
                    self._condition.notify_all()
                continue

            LOGGER.warning("Live stream read interrupted; reconnecting")
            self._capture.release()
            if not self._url:
                with self._condition:
                    self._stopped = True
                    self._condition.notify_all()
                return
            try:
                time.sleep(1.0)
                self._capture = get_youtube_stream(self._url, self._quality)
            except (RuntimeError, cv2.error) as exc:
                LOGGER.warning("Live stream reconnect failed: %s", exc)
                time.sleep(2.0)

    def read(self):
        """Return the newest frame, waiting briefly only for the first frame."""
        with self._condition:
            if self._frame is None and not self._stopped:
                self._condition.wait(timeout=10.0)
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def isOpened(self) -> bool:
        return not self._stopped

    def get(self, property_id: int) -> float:
        """Proxy capture properties needed by the annotated video writer."""
        return self._capture.get(property_id)

    @property
    def fps(self) -> float:
        """Return the measured decoder throughput for the live source."""
        return self._fps

    def release(self) -> None:
        self._stopped = True
        self._capture.release()
        with self._condition:
            self._condition.notify_all()
        if self._reader.is_alive():
            self._reader.join(timeout=2.0)


# Backward-compatible name for callers using the previous API.
LatestFrameCapture = ThreadedStreamLoader
