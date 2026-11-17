"""Minimalist modern white-theme web frontend and API server for Zebra Line Encroachment Penalty System."""
from __future__ import annotations

import csv
import json
import logging
import mimetypes
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None   # type: ignore

try:
    import config
    LOG_JSON_PATH = config.LOG_JSON_PATH
    LOG_CSV_PATH = config.LOG_CSV_PATH
except Exception:
    LOG_JSON_PATH = "violations/violations.json"
    LOG_CSV_PATH = "violations/violations.csv"

LOGGER = logging.getLogger("stripeguard.web")


# Global thread-safe state container
class DashboardState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_frame_jpeg: bytes | None = None
        self.signal_state: str = "GREEN"
        self.tracked_count: int = 0
        self.violation_count: int = 0
        self.fps: float = 0.0
        self.source_fps: float = 0.0
        self.vision_mode: str = "Clear Vision"
        self.signal_mode: str = "auto"
        self.manual_signal: str = "RED"
        self.recent_violations: list[dict[str, Any]] = []
        self.start_time: float = time.time()

    def update_frame(
        self,
        frame: Any,
        signal: str,
        tracked_count: int,
        violation_count: int,
        fps: float,
        source_fps: float = 0.0,
        vision_mode: str = "Clear Vision",
    ) -> None:
        jpeg_bytes: bytes | None = None
        if frame is not None:
            if cv2 is not None and hasattr(frame, "shape"):
                success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if success:
                    jpeg_bytes = encoded.tobytes()
            elif isinstance(frame, bytes):
                jpeg_bytes = frame

        with self.lock:
            if jpeg_bytes is not None:
                self.latest_frame_jpeg = jpeg_bytes
            self.signal_state = signal
            self.tracked_count = tracked_count
            self.violation_count = violation_count
            self.fps = round(fps, 1)
            self.vision_mode = vision_mode
            if source_fps > 0:
                self.source_fps = round(source_fps, 1)
                self.source_fps = round(source_fps, 1)

    def add_violation(self, entry: dict[str, Any]) -> None:
        with self.lock:
            v_id = entry.get("vehicle_id")
            v_time = entry.get("timestamp")
            for idx, existing in enumerate(self.recent_violations):
                if existing.get("vehicle_id") == v_id and existing.get("timestamp") == v_time:
                    self.recent_violations[idx] = entry
                    return
            self.recent_violations.insert(0, entry)
            self.recent_violations = self.recent_violations[:60]
            self.violation_count += 1

    def set_signal_override(self, mode: str, signal: str = "RED") -> None:
        with self.lock:
            self.signal_mode = mode
            if signal in ("RED", "GREEN", "YELLOW"):
                self.manual_signal = signal
                self.signal_state = signal
        try:
            override_path = Path("violations/signal_override.json")
            if mode == "auto":
                if override_path.exists():
                    override_path.unlink()
            else:
                override_path.parent.mkdir(parents=True, exist_ok=True)
                override_path.write_text(json.dumps({"mode": mode, "signal": signal}), encoding="utf-8")
        except Exception:
            pass

    def get_signal_override(self) -> tuple[str, str]:
        try:
            override_path = Path("violations/signal_override.json")
            if override_path.exists():
                data = json.loads(override_path.read_text(encoding="utf-8"))
                return data.get("mode", "auto"), data.get("signal", "RED")
        except Exception:
            pass
        with self.lock:
            return self.signal_mode, self.manual_signal

    def get_status_summary(self) -> dict[str, Any]:
        with self.lock:
            return {
                "signal": self.signal_state,
                "tracked_vehicles": self.tracked_count,
                "violation_count": self.violation_count,
                "fps": self.fps,
                "source_fps": self.source_fps,
                "vision_mode": self.vision_mode,
                "signal_mode": self.signal_mode,
                "manual_signal": self.manual_signal,
                "uptime_seconds": int(time.time() - self.start_time),
            }

    def load_initial_violations(self) -> None:
        try:
            if Path(LOG_JSON_PATH).exists():
                with open(LOG_JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        with self.lock:
                            self.recent_violations = list(reversed(data))[:60]
                            self.violation_count = len(data)
        except Exception:
            LOGGER.exception("Failed loading initial violations log")


STATE = DashboardState()


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>StripeGuard AI | Zebra Line Encroachment Penalty System</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 48'><rect width='48' height='48' rx='12' fill='%230f172a'/><path d='M10 16h28M10 24h28M10 32h28' stroke='white' stroke-width='3.2' stroke-linecap='round'/><circle cx='24' cy='24' r='14' stroke='%2338bdf8' stroke-width='2.4' stroke-dasharray='3 3' fill='none'/><circle cx='24' cy='24' r='4' fill='%23ef4444'/></svg>">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface-subtle: #f1f5f9;
      --border: #e2e8f0;
      --border-subtle: #edf2f7;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --accent: #2563eb;
      --danger: #dc2626;
      --danger-bg: #fef2f2;
      --success: #16a34a;
      --success-bg: #f0fdf4;
      --warning: #d97706;
      --warning-bg: #fffbeb;
      --radius-lg: 22px;
      --radius-md: 16px;
      --radius-sm: 10px;
      --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 20px -2px rgba(15, 23, 42, 0.06);
      --shadow-lg: 0 12px 32px -4px rgba(15, 23, 42, 0.08);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    body {
      background-color: var(--bg);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      -webkit-font-smoothing: antialiased;
    }

    /* Top Navigation Header */
    header {
      position: sticky;
      top: 0;
      z-index: 50;
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border);
      padding: 16px 36px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: var(--shadow-sm);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
      text-decoration: none;
      color: inherit;
    }
    .brand-icon {
      width: 42px;
      height: 42px;
      border-radius: var(--radius-sm);
      background: #0f172a;
      color: #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: var(--shadow-sm);
      flex-shrink: 0;
    }
    .brand-icon svg {
      width: 22px;
      height: 22px;
      stroke: #ffffff;
      stroke-width: 2.2;
    }
    .brand-text h1 {
      font-size: 18px;
      font-weight: 800;
      letter-spacing: -0.4px;
      color: var(--text-main);
    }
    .brand-text p {
      font-size: 12px;
      color: var(--text-muted);
      font-weight: 500;
    }

    /* Status Pills in Header */
    .status-group {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 16px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 700;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-main);
      box-shadow: var(--shadow-sm);
      transition: all 0.25s ease;
    }
    .pill-signal {
      padding: 8px 18px;
      letter-spacing: 0.3px;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
    }
    .signal-red {
      color: var(--danger);
      background: var(--danger-bg);
      border-color: #fecaca;
    }
    .signal-red .dot {
      background: var(--danger);
      animation: pulse 1.4s infinite;
    }
    .signal-green {
      color: var(--success);
      background: var(--success-bg);
      border-color: #bbf7d0;
    }
    .signal-green .dot {
      background: var(--success);
    }
    .signal-yellow {
      color: var(--warning);
      background: var(--warning-bg);
      border-color: #fef08a;
    }
    .signal-yellow .dot {
      background: var(--warning);
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.35; transform: scale(0.85); }
    }

    /* Main Dashboard Layout */
    main {
      flex: 1;
      padding: 28px 36px;
      display: grid;
      grid-template-columns: 1fr 440px;
      gap: 28px;
      max-width: 1720px;
      margin: 0 auto;
      width: 100%;
    }

    @media (max-width: 1180px) {
      main {
        grid-template-columns: 1fr;
        padding: 20px;
      }
      header {
        padding: 14px 20px;
      }
    }

    /* Left: Incident Spotlight & Telemetry Panel */
    .spotlight-section {
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .spotlight-card {
      background: var(--surface);
      border-radius: var(--radius-lg);
      border: 1px solid var(--border);
      padding: 24px;
      box-shadow: var(--shadow-md);
      display: flex;
      flex-direction: column;
      gap: 18px;
      position: relative;
    }

    .spotlight-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }
    .spotlight-title {
      font-size: 14px;
      font-weight: 800;
      letter-spacing: -0.2px;
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text-main);
    }
    .spotlight-pulse {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--danger);
      animation: pulse 1.4s infinite;
    }

    .spotlight-body {
      display: grid;
      grid-template-columns: 1.15fr 1fr;
      gap: 22px;
      align-items: center;
    }

    @media (max-width: 800px) {
      .spotlight-body {
        grid-template-columns: 1fr;
      }
    }

    .spotlight-img-box {
      width: 100%;
      height: 290px;
      background: #0f172a;
      border-radius: var(--radius-md);
      overflow: hidden;
      border: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    .spotlight-img-box img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      image-rendering: -webkit-optimize-contrast;
    }

    .spotlight-meta {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .meta-box {
      background: var(--surface-subtle);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 12px 14px;
    }
    .meta-label {
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
    }
    .meta-value {
      font-size: 15px;
      font-weight: 800;
      color: var(--text-main);
    }

    /* Status Bar */
    .mode-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 14px 24px;
      box-shadow: var(--shadow-sm);
      gap: 12px;
      flex-wrap: wrap;
    }

    .mode-indicator {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 13px;
      font-weight: 600;
      color: var(--text-main);
    }
    .mode-badge {
      background: var(--success-bg);
      color: var(--success);
      border: 1px solid #bbf7d0;
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.3px;
    }

    .btn-export {
      background: var(--surface-subtle);
      border: 1px solid var(--border);
      color: var(--text-main);
      padding: 9px 18px;
      border-radius: var(--radius-sm);
      font-size: 13px;
      font-weight: 600;
      text-decoration: none;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s ease;
    }
    .btn-export:hover {
      background: #e2e8f0;
      transform: translateY(-1px);
    }

    /* Right: Violation Photo Feed */
    .violation-section {
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-height: calc(100vh - 120px);
    }

    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 4px;
    }
    .panel-title {
      font-size: 16px;
      font-weight: 800;
      letter-spacing: -0.3px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .count-chip {
      font-size: 12px;
      background: var(--danger-bg);
      color: var(--danger);
      border: 1px solid #fecaca;
      padding: 3px 10px;
      border-radius: 9999px;
      font-weight: 800;
    }

    .feed-container {
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
      padding-right: 6px;
    }

    /* Clean custom scrollbar */
    .feed-container::-webkit-scrollbar {
      width: 6px;
    }
    .feed-container::-webkit-scrollbar-thumb {
      background: #cbd5e1;
      border-radius: 10px;
    }

    .violation-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 14px;
      display: flex;
      gap: 16px;
      align-items: center;
      transition: all 0.2s ease;
      cursor: pointer;
      box-shadow: var(--shadow-sm);
      position: relative;
    }
    .violation-card:hover {
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
      border-color: #cbd5e1;
    }
    .violation-card::before {
      content: "";
      position: absolute;
      left: 0;
      top: 14px;
      bottom: 14px;
      width: 4px;
      border-radius: 0 4px 4px 0;
      background: var(--danger);
    }

    .thumb-wrapper {
      width: 88px;
      height: 68px;
      border-radius: var(--radius-sm);
      overflow: hidden;
      background: #0f172a;
      flex-shrink: 0;
      border: 1px solid var(--border);
    }
    .thumb-wrapper img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
      transition: transform 0.25s ease;
    }
    .violation-card:hover .thumb-wrapper img {
      transform: scale(1.06);
    }

    .violation-info {
      flex: 1;
      min-width: 0;
    }
    .info-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }
    .vehicle-tag {
      font-size: 14px;
      font-weight: 800;
      color: var(--text-main);
    }
    .time-stamp {
      font-size: 12px;
      color: var(--text-muted);
      font-weight: 500;
    }
    .plate-badge {
      display: inline-block;
      font-size: 11px;
      font-weight: 800;
      font-family: monospace, sans-serif;
      padding: 4px 9px;
      border-radius: 6px;
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      color: var(--accent);
      letter-spacing: 0.5px;
    }

    .empty-state {
      padding: 48px 24px;
      text-align: center;
      color: var(--text-muted);
      font-size: 13px;
      background: var(--surface);
      border: 1px dashed var(--border);
      border-radius: var(--radius-md);
      font-weight: 500;
    }

    /* Modal / Lightbox for Evidence */
    .modal {
      display: none;
      position: fixed;
      inset: 0;
      z-index: 100;
      background: rgba(15, 23, 42, 0.6);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .modal-content {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      max-width: 660px;
      width: 100%;
      overflow: hidden;
      box-shadow: 0 25px 50px -12px rgba(15, 23, 42, 0.25);
      animation: modalIn 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes modalIn {
      from { opacity: 0; transform: scale(0.96); }
      to { opacity: 1; transform: scale(1); }
    }
    .modal-img-container {
      width: 100%;
      max-height: 400px;
      background: #020617;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 10px;
    }
    .modal-img-container img {
      max-width: 100%;
      max-height: 380px;
      object-fit: contain;
      border-radius: var(--radius-sm);
      image-rendering: -webkit-optimize-contrast;
      image-rendering: crisp-edges;
    }
    .modal-body {
      padding: 22px 26px;
    }
    .modal-footer {
      padding: 16px 26px;
      border-top: 1px solid var(--border);
      display: flex;
      justify-content: flex-end;
      gap: 12px;
      background: var(--surface-subtle);
    }
    .btn-close {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-main);
      padding: 8px 16px;
      border-radius: var(--radius-sm);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
    }
    .btn-download {
      background: var(--accent);
      border: 1px solid var(--accent);
      color: #ffffff;
      padding: 8px 18px;
      border-radius: var(--radius-sm);
      font-size: 13px;
      font-weight: 700;
      text-decoration: none;
      box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
    }

    .sound-btn {
      width: 38px;
      height: 38px;
      border-radius: 9999px;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-muted);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: var(--shadow-sm);
      transition: all 0.2s;
    }
    .sound-btn:hover {
      color: var(--text-main);
      border-color: #cbd5e1;
    }
  </style>
</head>
<body>

  <!-- Top Navigation Header -->
  <header>
    <a href="/" class="brand">
      <div class="brand-icon">
        <svg viewBox="0 0 32 32" fill="none">
          <rect width="32" height="32" rx="8" fill="#0f172a"/>
          <!-- Road with zebra crossing markings -->
          <path d="M7 10h18M7 16h18M7 22h18" stroke="#ffffff" stroke-width="2.4" stroke-linecap="round"/>
          <!-- AI Surveillance Radar Reticle -->
          <circle cx="16" cy="16" r="10" stroke="#38bdf8" stroke-width="1.8" stroke-dasharray="2.5 2.5" fill="none"/>
          <circle cx="16" cy="16" r="3.5" fill="#ef4444"/>
        </svg>
      </div>
      <div class="brand-text">
        <h1>StripeGuard AI</h1>
        <p>Zebra Line Encroachment Penalty System</p>
      </div>
    </a>

    <!-- Status Pills -->
    <div class="status-group">
      <div class="pill">
        <span style="color:var(--danger); font-size:14px;">&#9888;</span>
        <span id="violationCountPill">0 Violations</span>
      </div>
      <div class="pill" id="visionPill" style="background:#eff6ff; border-color:#bfdbfe; color:#1d4ed8;" title="Adaptive Night & Adverse Weather Vision Engine">
        <span id="visionIcon">☀️</span>
        <span id="visionModeText">Adaptive Vision Active</span>
      </div>
      <div class="pill">
        <span style="color:#7c3aed; font-size:14px;">&#9889;</span>
        <span id="fpsText">0.0 FPS</span>
      </div>
      <button id="soundToggle" class="sound-btn" title="Toggle audio chime">
        <svg id="soundIcon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M11 5L6 9H2v6h4l5 4V5z"/>
          <path d="M19.07 4.93a10 10 0 010 14.14M15.54 8.46a5 5 0 010 7.07"/>
        </svg>
      </button>
    </div>
  </header>

  <!-- Main View -->
  <main>
    <!-- Left Column: Spotlight Incident & Telemetry Panel -->
    <section class="spotlight-section">
      <div class="spotlight-card">
        <div class="spotlight-header">
          <div class="spotlight-title">
            <span class="spotlight-pulse"></span>
            <span>LATEST CAPTURED INCIDENT SPOTLIGHT</span>
          </div>
          <span id="spotlightTimeTag" style="font-size:12px; font-weight:700; color:var(--text-muted);">Real-Time Enforcement</span>
        </div>

        <div id="spotlightContent" class="spotlight-body">
          <div class="spotlight-img-box">
            <img id="spotlightImg" src="" alt="Latest Vehicle Evidence" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 24 24%22 fill=%22%23cbd5e1%22><text x=%2250%%22 y=%2250%%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 fill=%22white%22 font-size=%228%22>PHOTO</text></svg>'"/>
          </div>
          <div class="spotlight-meta">
            <div class="meta-box">
              <div class="meta-label">Vehicle Identifier &amp; Infraction</div>
              <div id="spotlightVehicleId" class="meta-value" style="color:var(--danger); font-size:16px;">Vehicle #-- (Zebra Encroachment)</div>
            </div>
            <div class="meta-box">
              <div class="meta-label">License Plate (ANPR)</div>
              <div style="display:flex; align-items:center; gap:10px; margin-top:4px;">
                <span id="spotlightPlateText" class="plate-badge" style="font-size:13px; padding:5px 12px;">AWAITING...</span>
                <span style="font-size:11px; font-weight:800; color:var(--danger); background:var(--danger-bg); border:1px solid #fecaca; padding:3px 8px; border-radius:9999px;">RED ENCROACH</span>
              </div>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
              <div class="meta-box">
                <div class="meta-label">OCR Confidence</div>
                <div id="spotlightConf" class="meta-value" style="color:var(--success);">--%</div>
              </div>
              <div class="meta-box">
                <div class="meta-label">Capture Time</div>
                <div id="spotlightTime" class="meta-value">--:--:--</div>
              </div>
            </div>
            <button class="btn-download" style="text-align:center; padding:10px 18px; cursor:pointer; width:100%; border:none;" onclick="openSpotlightModal()">
              Inspect Full Evidence &amp; Download JPG
            </button>
          </div>
        </div>

        <div id="spotlightEmpty" style="display:none; padding:56px 24px; text-align:center; color:var(--text-muted); flex-direction:column; align-items:center; gap:12px;">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" style="color:#94a3b8;">
            <circle cx="12" cy="12" r="10"/>
            <path d="M12 6v6l4 2"/>
          </svg>
          <div style="font-size:16px; font-weight:800; color:var(--text-main);">Active Radar Surveillance Active</div>
          <p style="font-size:13px; max-width:440px; line-height:1.5;">Monitoring pedestrian zebra crossings. Any vehicle encroaching on RED will be captured, enhanced, and presented here in real time.</p>
        </div>
      </div>

      <!-- Telemetry & Mode Bar -->
      <div class="mode-bar">
        <div class="mode-indicator">
          <button id="signalControlBtn" onclick="toggleSignalOverride()" class="pill pill-signal signal-green" style="cursor:pointer; border:none; outline:none; font-size:12px; font-weight:800; padding:6px 14px;" title="Click to toggle traffic light state">
            <span class="dot"></span>
            <span id="signalControlText">GREEN LIGHT (LEGAL)</span>
          </button>
          <span style="font-size:12px; color:var(--text-muted); font-weight:500;">(Click light to toggle RED / GREEN)</span>
        </div>
        <a href="/api/export_csv" download="violations.csv" class="btn-export">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
          </svg>
          Export CSV Audit Log
        </a>
      </div>
    </section>

    <!-- Right Column: Violations Feed -->
    <aside class="violation-section">
      <div class="panel-header">
        <div class="panel-title">
          <span>Violation Evidence Log</span>
          <span id="violationCounterChip" class="count-chip">0</span>
        </div>
        <span style="font-size:12px; color:var(--text-muted); font-weight:500;">Real-Time Capture</span>
      </div>

      <div id="violationFeed" class="feed-container">
        <div class="empty-state">No violations recorded.<br>Monitoring pedestrian zebra zone...</div>
      </div>
    </aside>
  </main>

  <!-- Evidence Lightbox Modal -->
  <div id="evidenceModal" class="modal" onclick="closeModal(event)">
    <div class="modal-content" onclick="event.stopPropagation()">
      <div class="modal-img-container">
        <img id="modalImg" src="" alt="Violation Snapshot" />
      </div>
      <div class="modal-body">
        <div id="modalPlateImgContainer" style="display:none; margin-bottom:16px; background:#f8fafc; padding:12px; border-radius:var(--radius-sm); border:1px solid var(--border); text-align:center;">
          <div style="font-size:11px; color:var(--text-muted); margin-bottom:6px; letter-spacing:0.5px; font-weight:700;">ENHANCED LICENSE PLATE CROP</div>
          <img id="modalPlateImg" src="" style="max-height:85px; max-width:100%; object-fit:contain; border-radius:6px;" alt="Enhanced Plate" />
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
          <h2 id="modalTitle" style="font-size:18px; font-weight:800; color:var(--text-main);">Vehicle #ID-0</h2>
          <span id="modalTime" style="font-size:13px; color:var(--text-muted); font-weight:500;">--:--:--</span>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; font-size:12px;">
          <div style="background:var(--surface-subtle); padding:12px; border-radius:var(--radius-sm); border:1px solid var(--border);">
            <div style="color:var(--text-muted); margin-bottom:4px; font-weight:600;">License Plate (ANPR)</div>
            <div id="modalPlate" style="font-size:15px; font-weight:800; color:var(--accent); font-family:monospace;">UNKNOWN</div>
          </div>
          <div style="background:var(--surface-subtle); padding:12px; border-radius:var(--radius-sm); border:1px solid var(--border);">
            <div style="color:var(--text-muted); margin-bottom:4px; font-weight:600;">OCR Confidence</div>
            <div id="modalConfidence" style="font-size:15px; font-weight:800; color:var(--success);">0.0%</div>
          </div>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn-close" onclick="document.getElementById('evidenceModal').style.display='none'">Close</button>
        <a id="modalDownload" href="#" download class="btn-download">Download Evidence JPG</a>
      </div>
    </div>
  </div>

  <script>
    let soundEnabled = true;
    let lastViolationCount = 0;
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    function playAlertTone() {
      if (!soundEnabled) return;
      try {
        if (audioCtx.state === 'suspended') audioCtx.resume();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(880, audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.25);
        gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.25);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.25);
      } catch (e) {
        console.warn('Audio play error', e);
      }
    }

    document.getElementById('soundToggle').addEventListener('click', () => {
      soundEnabled = !soundEnabled;
      document.getElementById('soundToggle').style.opacity = soundEnabled ? '1' : '0.4';
    });

    let currentSignalState = 'RED';

    async function toggleSignalOverride() {
      const nextSignal = currentSignalState === 'RED' ? 'GREEN' : 'RED';
      try {
        await fetch('/api/signal', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ mode: 'manual', signal: nextSignal })
        });
        currentSignalState = nextSignal;
        updateSignalBtn(nextSignal);
      } catch (err) {
        console.error('Signal toggle error:', err);
      }
    }

    function updateSignalBtn(sig) {
      const btn = document.getElementById('signalControlBtn');
      const text = document.getElementById('signalControlText');
      const badge = document.getElementById('modeBadge');
      if (!btn || !text) return;
      if (sig === 'RED') {
        btn.className = 'pill pill-signal signal-red';
        text.innerText = 'RED LIGHT (ACTIVE)';
        if (badge) {
          badge.innerText = 'ENFORCING';
          badge.style.background = 'var(--danger-bg)';
          badge.style.color = 'var(--danger)';
          badge.style.borderColor = '#fecaca';
        }
      } else {
        btn.className = 'pill pill-signal signal-green';
        text.innerText = 'GREEN LIGHT (LEGAL)';
        if (badge) {
          badge.innerText = 'MONITORING';
          badge.style.background = 'var(--success-bg)';
          badge.style.color = 'var(--success)';
          badge.style.borderColor = '#bbf7d0';
        }
      }
    }

    async function fetchStatus() {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();
        const vPill = document.getElementById('violationCountPill');
        if (vPill) vPill.innerText = data.violation_count + ' Violations';
        const fpsEl = document.getElementById('fpsText');
        if (fpsEl) fpsEl.innerText = data.fps.toFixed(1) + ' FPS';
        if (data.signal && data.signal !== currentSignalState) {
          currentSignalState = data.signal;
          updateSignalBtn(data.signal);
        }
        if (data.vision_mode) {
          const vText = document.getElementById('visionModeText');
          const vIcon = document.getElementById('visionIcon');
          const vP = document.getElementById('visionPill');
          if (vText) vText.innerText = data.vision_mode;
          if (data.vision_mode.includes('Night')) {
            if (vIcon) vIcon.innerText = '🌙';
            if (vP) { vP.style.background = '#312e81'; vP.style.color = '#e0e7ff'; vP.style.borderColor = '#4338ca'; }
          } else if (data.vision_mode.includes('Rain') || data.vision_mode.includes('Adverse')) {
            if (vIcon) vIcon.innerText = '🌧️';
            if (vP) { vP.style.background = '#0e7490'; vP.style.color = '#cffafe'; vP.style.borderColor = '#155e75'; }
          } else {
            if (vIcon) vIcon.innerText = '☀️';
            if (vP) { vP.style.background = '#eff6ff'; vP.style.color = '#1d4ed8'; vP.style.borderColor = '#bfdbfe'; }
          }
        }
      } catch (err) {
        console.warn('Status poll error', err);
      }
    }

    let latestSpotlightItem = null;

    function openSpotlightModal() {
      if (!latestSpotlightItem) return;
      const cleanPath = (latestSpotlightItem.file_path || '').replace(/\\\\/g, '/');
      const cleanPlatePath = (latestSpotlightItem.plate_file_path || '').replace(/\\\\/g, '/');
      const time = new Date(latestSpotlightItem.timestamp).toLocaleTimeString();
      const conf = (parseFloat(latestSpotlightItem.confidence || 0) * 100).toFixed(0);
      openModal(cleanPath, latestSpotlightItem.vehicle_id, time, latestSpotlightItem.plate_number, conf, cleanPlatePath);
    }

    async function fetchViolations() {
      try {
        const res = await fetch('/api/violations');
        if (!res.ok) return;
        const list = await res.json();
        
        document.getElementById('violationCounterChip').innerText = list.length;
        const feed = document.getElementById('violationFeed');

        if (list.length > lastViolationCount && lastViolationCount > 0) {
          playAlertTone();
        }
        lastViolationCount = list.length;

        if (list.length === 0) {
          feed.innerHTML = '<div class="empty-state">No violations recorded.<br>Monitoring pedestrian zebra zone...</div>';
          document.getElementById('spotlightContent').style.display = 'none';
          document.getElementById('spotlightEmpty').style.display = 'flex';
          return;
        }

        const top = list[0];
        document.getElementById('spotlightContent').style.display = 'grid';
        document.getElementById('spotlightEmpty').style.display = 'none';
        const topPath = (top.file_path || '').replace(/\\\\/g, '/');
        document.getElementById('spotlightImg').src = '/' + topPath + '?t=' + encodeURIComponent(top.timestamp || '');
        document.getElementById('spotlightVehicleId').innerText = 'Vehicle #' + top.vehicle_id + ' (Zebra Encroachment)';
        document.getElementById('spotlightPlateText').innerText = top.plate_number || 'PENDING';
        document.getElementById('spotlightConf').innerText = (parseFloat(top.confidence || 0) * 100).toFixed(0) + '%';
        const topTime = new Date(top.timestamp).toLocaleTimeString();
        document.getElementById('spotlightTime').innerText = topTime;
        latestSpotlightItem = top;

        let html = '';
        for (const item of list) {
          const id = item.vehicle_id;
          const plate = item.plate_number || 'PENDING';
          const time = new Date(item.timestamp).toLocaleTimeString();
          const cleanPath = (item.file_path || '').replace(/\\\\/g, '/');
          const cleanPlatePath = (item.plate_file_path || '').replace(/\\\\/g, '/');
          const conf = (parseFloat(item.confidence || 0) * 100).toFixed(0);

          html += `
            <div class="violation-card" onclick="openModal('${cleanPath}', '${id}', '${time}', '${plate}', '${conf}', '${cleanPlatePath}')">
              <div class="thumb-wrapper">
                <img src="/${cleanPath}?t=${encodeURIComponent(item.timestamp || '')}" alt="ID ${id}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 24 24%22 fill=%22%23cbd5e1%22><text x=%2250%%22 y=%2250%%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 fill=%22white%22 font-size=%2210%22>PHOTO</text></svg>'"/>
              </div>
              <div class="violation-info">
                <div class="info-top">
                  <span class="vehicle-tag">Vehicle #${id}</span>
                  <span class="time-stamp">${time}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <span class="plate-badge">${plate}</span>
                  <span style="font-size:11px; color:var(--danger); font-weight:800;">RED ENCROACH</span>
                </div>
              </div>
            </div>
          `;
        }
        feed.innerHTML = html;
      } catch (err) {
        console.warn('Violations poll error', err);
      }
    }

    function openModal(imgSrc, id, time, plate, conf, plateImg) {
      document.getElementById('modalImg').src = '/' + imgSrc;
      document.getElementById('modalTitle').innerText = 'Vehicle #' + id + ' (Zebra Violation)';
      document.getElementById('modalTime').innerText = time;
      document.getElementById('modalPlate').innerText = plate;
      document.getElementById('modalConfidence').innerText = conf + '%';
      document.getElementById('modalDownload').href = '/' + imgSrc;
      
      const plateContainer = document.getElementById('modalPlateImgContainer');
      if (plateImg && plateImg !== '' && plateImg !== 'undefined') {
        document.getElementById('modalPlateImg').src = '/' + plateImg;
        plateContainer.style.display = 'block';
      } else {
        plateContainer.style.display = 'none';
      }
      document.getElementById('evidenceModal').style.display = 'flex';
    }

    function closeModal(e) {
      if (e.target.id === 'evidenceModal') {
        document.getElementById('evidenceModal').style.display = 'none';
      }
    }

    // Polling intervals
    setInterval(fetchStatus, 400);
    setInterval(fetchViolations, 1200);
    fetchStatus();
    fetchViolations();
  </script>
</body>
</html>
"""


class DashboardHTTPHandler(BaseHTTPRequestHandler):
    """HTTP Request handler serving the white-theme web dashboard and live stream."""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy HTTP logging in console
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        # 1. Main Dashboard UI
        if path in ("/", "/index.html"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        # 2. Live MJPEG Video Stream
        elif path == "/video_feed":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()

            live_file = Path("violations/live_frame.jpg")
            try:
                while True:
                    frame_bytes = None
                    with STATE.lock:
                        frame_bytes = STATE.latest_frame_jpeg

                    if frame_bytes is None and live_file.exists():
                        try:
                            frame_bytes = live_file.read_bytes()
                        except Exception:
                            pass

                    if frame_bytes is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode("utf-8"))
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.033)  # ~30 FPS streaming
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            return

        # 3. Status API
        elif path == "/api/status":
            summary = STATE.get_status_summary()
            try:
                status_file = Path("violations/status.json")
                if status_file.exists():
                    with open(status_file, "r", encoding="utf-8") as f:
                        file_status = json.load(f)
                        if isinstance(file_status, dict):
                            summary.update(file_status)
            except Exception:
                pass
            if STATE.signal_mode == "manual":
                summary["signal"] = STATE.manual_signal
                summary["signal_mode"] = "manual"
            try:
                if Path(LOG_JSON_PATH).exists():
                    with open(LOG_JSON_PATH, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            summary["violation_count"] = max(summary.get("violation_count", 0), len(data))
            except Exception:
                pass
            payload = json.dumps(summary).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(payload)
            return

        # 4. Violations List API
        elif path == "/api/violations":
            violations: list[dict[str, Any]] = []
            try:
                if Path(LOG_JSON_PATH).exists():
                    with open(LOG_JSON_PATH, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            violations = list(reversed(data))[:60]
            except Exception:
                pass

            if not violations:
                with STATE.lock:
                    violations = list(STATE.recent_violations)[:60]

            payload = json.dumps(violations).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(payload)
            return

        # 5. Export CSV API
        elif path == "/api/export_csv":
            csv_path = Path(LOG_CSV_PATH)
            if csv_path.exists():
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", 'attachment; filename="violations.csv"')
                self.end_headers()
                with open(csv_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "CSV Log not found")
            return

        # 6. Serve static violation snapshot images (/violations/<filename>)
        elif path.startswith("/violations/"):
            rel_file = path.lstrip("/")
            file_path = Path(rel_file)
            if file_path.exists() and file_path.is_file():
                mime, _ = mimetypes.guess_type(str(file_path))
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime or "image/jpeg")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Snapshot not found")
            return

        # Fallback 404
        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/signal":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8").strip()
            mode = "auto"
            signal = "RED"
            try:
                if body.startswith("{"):
                    data = json.loads(body)
                    mode = data.get("mode", "auto")
                    signal = data.get("signal", "RED")
                else:
                    params = parse_qs(body)
                    mode = params.get("mode", ["auto"])[0]
                    signal = params.get("signal", ["RED"])[0]
            except Exception:
                pass
            STATE.set_signal_override(mode, signal)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
            return

        self.send_error(HTTPStatus.NOT_FOUND, "POST endpoint not found")


class WebDashboard:
    """Multi-threaded background server wrapper for StripeGuard AI."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        self.host = host
        self.port = port
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        STATE.load_initial_violations()
        try:
            self.server = ThreadingHTTPServer((self.host, self.port), DashboardHTTPHandler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            print(f"\n=======================================================")
            print(f"  >>> Zebra Line Encroachment Penalty System Running! <<<")
            print(f"  Open your browser: http://localhost:{self.port}")
            print(f"=======================================================\n")
        except OSError as e:
            if getattr(e, "winerror", None) == 10048 or "Address already in use" in str(e):
                print(f"[INFO] Web dashboard is already active on http://localhost:{self.port}")
            else:
                raise

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()


# Module-level convenience functions
_GLOBAL_DASHBOARD: WebDashboard | None = None

def start_dashboard(host: str = "0.0.0.0", port: int = 5000) -> WebDashboard:
    global _GLOBAL_DASHBOARD
    if _GLOBAL_DASHBOARD is None:
        _GLOBAL_DASHBOARD = WebDashboard(host, port)
        _GLOBAL_DASHBOARD.start()
    return _GLOBAL_DASHBOARD

def update_dashboard(
    frame: Any,
    signal: str,
    tracked_count: int,
    violation_count: int,
    fps: float,
    source_fps: float = 0.0,
    vision_mode: str = "Clear Vision",
) -> None:
    STATE.update_frame(frame, signal, tracked_count, violation_count, fps, source_fps, vision_mode)
    try:
        if frame is not None and cv2 is not None and hasattr(frame, "shape"):
            live_path = Path("violations/live_frame.jpg")
            live_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(live_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        status_path = Path("violations/status.json")
        status_path.write_text(
            json.dumps({
                "signal": signal,
                "tracked_vehicles": tracked_count,
                "violation_count": violation_count,
                "fps": round(fps, 1),
                "source_fps": round(source_fps, 1),
                "vision_mode": vision_mode,
            }),
            encoding="utf-8",
        )
    except Exception:
        pass

def add_violation_to_dashboard(entry: dict[str, Any]) -> None:
    STATE.add_violation(entry)

def get_signal_override() -> tuple[str, str]:
    return STATE.get_signal_override()


if __name__ == "__main__":
    server = start_dashboard(port=5000)
    print("Dashboard started in standalone mode. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        print("\nDashboard stopped.")
