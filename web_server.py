"""Minimalist modern web frontend and API server for StripeGuard AI."""
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
        self.signal_mode: str = "auto"  # 'auto' or 'manual'
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
            if source_fps > 0:
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

    def get_signal_override(self) -> tuple[str, str]:
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
  <title>StripeGuard AI &mdash; Intelligent Traffic Enforcement</title>
  <style>
    :root {
      --bg: #080c14;
      --card-bg: rgba(255, 255, 255, 0.035);
      --card-border: rgba(255, 255, 255, 0.07);
      --card-hover: rgba(255, 255, 255, 0.06);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #3b82f6;
      --danger: #ef4444;
      --success: #10b981;
      --warning: #f59e0b;
      --radius: 18px;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Helvetica, Arial, sans-serif;
    }

    body {
      background-color: var(--bg);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }

    /* Ambient background glow */
    .glow-sphere {
      position: fixed;
      width: 450px;
      height: 450px;
      border-radius: 50%;
      filter: blur(140px);
      pointer-events: none;
      z-index: 0;
      opacity: 0.15;
      transition: background-color 0.8s ease, transform 0.8s ease;
    }
    .glow-left { top: -100px; left: -100px; background-color: var(--accent); }
    .glow-right { bottom: -100px; right: -100px; background-color: var(--danger); }

    /* Top Navigation Header */
    header {
      position: sticky;
      top: 0;
      z-index: 50;
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      background: rgba(8, 12, 20, 0.75);
      border-bottom: 1px solid var(--card-border);
      padding: 14px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      text-decoration: none;
      color: inherit;
    }
    .brand-logo {
      width: 34px;
      height: 34px;
      border-radius: 10px;
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4);
    }
    .brand-logo svg {
      width: 20px;
      height: 20px;
      stroke: #ffffff;
      stroke-width: 2.2;
    }
    .brand-text h1 {
      font-size: 17px;
      font-weight: 700;
      letter-spacing: -0.3px;
    }
    .brand-text p {
      font-size: 11px;
      color: var(--text-muted);
      letter-spacing: 0.2px;
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
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      transition: all 0.3s ease;
    }
    .pill-signal {
      padding: 7px 16px;
      font-weight: 700;
      letter-spacing: 0.5px;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }
    .signal-red {
      color: #fca5a5;
      border-color: rgba(239, 68, 68, 0.4);
      background: rgba(239, 68, 68, 0.12);
      box-shadow: 0 0 16px rgba(239, 68, 68, 0.25);
    }
    .signal-red .dot {
      background: var(--danger);
      box-shadow: 0 0 8px var(--danger);
      animation: pulse 1.4s infinite;
    }
    .signal-green {
      color: #86efac;
      border-color: rgba(16, 185, 129, 0.4);
      background: rgba(16, 185, 129, 0.12);
      box-shadow: 0 0 16px rgba(16, 185, 129, 0.25);
    }
    .signal-green .dot {
      background: var(--success);
      box-shadow: 0 0 8px var(--success);
    }
    .signal-yellow {
      color: #fde047;
      border-color: rgba(245, 158, 11, 0.4);
      background: rgba(245, 158, 11, 0.12);
    }
    .signal-yellow .dot { background: var(--warning); }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Main Dashboard Layout */
    main {
      position: relative;
      z-index: 10;
      flex: 1;
      padding: 24px 28px;
      display: grid;
      grid-template-columns: 1fr 410px;
      gap: 24px;
      max-width: 1780px;
      margin: 0 auto;
      width: 100%;
    }

    @media (max-width: 1100px) {
      main {
        grid-template-columns: 1fr;
      }
    }

    /* Left: Live Stream Panel */
    .stream-section {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .video-card {
      position: relative;
      background: #000000;
      border-radius: var(--radius);
      border: 1px solid var(--card-border);
      overflow: hidden;
      box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.7);
      aspect-ratio: 16 / 9;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .video-card img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    /* Stream Overlays */
    .stream-overlay-top {
      position: absolute;
      top: 14px;
      left: 14px;
      right: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      pointer-events: none;
    }
    .overlay-badge {
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      background: rgba(0, 0, 0, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.12);
      padding: 5px 12px;
      border-radius: 8px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.4px;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .rec-indicator {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #ef4444;
      animation: pulse 1.2s infinite;
    }

    /* Stream Controls Bar */
    .controls-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 12px 20px;
    }

    .btn-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .btn {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      color: var(--text-main);
      padding: 8px 16px;
      border-radius: 10px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.2s ease;
    }
    .btn:hover {
      background: var(--card-hover);
      border-color: rgba(255, 255, 255, 0.16);
      transform: translateY(-1px);
    }
    .btn-active {
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
      box-shadow: 0 4px 12px rgba(59, 130, 246, 0.35);
    }
    .btn-danger-active {
      background: var(--danger);
      border-color: var(--danger);
      color: #ffffff;
      box-shadow: 0 4px 12px rgba(239, 68, 68, 0.35);
    }
    .btn-success-active {
      background: var(--success);
      border-color: var(--success);
      color: #ffffff;
      box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35);
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
      font-size: 15px;
      font-weight: 700;
      letter-spacing: -0.2px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .count-chip {
      font-size: 11px;
      background: rgba(239, 68, 68, 0.15);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.3);
      padding: 2px 8px;
      border-radius: 999px;
      font-weight: 700;
    }

    .feed-container {
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding-right: 4px;
    }

    /* Modern scrollbar */
    .feed-container::-webkit-scrollbar {
      width: 5px;
    }
    .feed-container::-webkit-scrollbar-thumb {
      background: rgba(255, 255, 255, 0.1);
      border-radius: 10px;
    }

    .violation-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 12px;
      display: flex;
      gap: 14px;
      align-items: center;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      cursor: pointer;
      position: relative;
    }
    .violation-card:hover {
      background: var(--card-hover);
      border-color: rgba(239, 68, 68, 0.3);
      transform: translateX(-3px);
      box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.5);
    }
    .violation-card::before {
      content: "";
      position: absolute;
      left: 0;
      top: 12px;
      bottom: 12px;
      width: 3px;
      border-radius: 0 4px 4px 0;
      background: var(--danger);
    }

    .thumb-wrapper {
      width: 82px;
      height: 62px;
      border-radius: 8px;
      overflow: hidden;
      background: #111827;
      flex-shrink: 0;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .thumb-wrapper img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
      transition: transform 0.3s ease;
    }
    .violation-card:hover .thumb-wrapper img {
      transform: scale(1.08);
    }

    .violation-info {
      flex: 1;
      min-width: 0;
    }
    .info-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 4px;
    }
    .vehicle-tag {
      font-size: 13px;
      font-weight: 700;
      color: #f1f5f9;
    }
    .time-stamp {
      font-size: 11px;
      color: var(--text-muted);
    }
    .plate-badge {
      display: inline-block;
      font-size: 11px;
      font-weight: 700;
      font-family: monospace, sans-serif;
      padding: 3px 8px;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.07);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: #38bdf8;
      letter-spacing: 0.5px;
    }

    .empty-state {
      padding: 40px 20px;
      text-align: center;
      color: var(--text-muted);
      font-size: 13px;
      border: 1px dashed var(--card-border);
      border-radius: var(--radius);
    }

    /* Fullscreen Modal / Lightbox for Violation Evidence */
    .modal {
      display: none;
      position: fixed;
      inset: 0;
      z-index: 100;
      background: rgba(4, 6, 12, 0.85);
      backdrop-filter: blur(16px);
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .modal-content {
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 20px;
      max-width: 680px;
      width: 100%;
      overflow: hidden;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
      animation: modalIn 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes modalIn {
      from { opacity: 0; transform: scale(0.95); }
      to { opacity: 1; transform: scale(1); }
    }
    .modal-img-container {
      width: 100%;
      max-height: 400px;
      background: #000000;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .modal-img-container img {
      max-width: 100%;
      max-height: 400px;
      object-fit: contain;
    }
    .modal-body {
      padding: 20px 24px;
    }
    .modal-footer {
      padding: 14px 24px;
      border-top: 1px solid var(--card-border);
      display: flex;
      justify-content: flex-end;
      gap: 12px;
      background: rgba(0, 0, 0, 0.2);
    }

    /* Sound alert notification icon */
    .sound-btn {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s;
    }
    .sound-btn:hover { color: var(--text-main); }
  </style>
</head>
<body>
  <div class="glow-sphere glow-left"></div>
  <div class="glow-sphere glow-right"></div>

  <!-- Header -->
  <header>
    <a href="/" class="brand">
      <div class="brand-logo">
        <svg viewBox="0 0 24 24" fill="none">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div class="brand-text">
        <h1>StripeGuard AI</h1>
        <p>Real-Time Zebra Line Enforcement System</p>
      </div>
    </a>

    <!-- Top Status Pills -->
    <div class="status-group">
      <div id="signalPill" class="pill pill-signal signal-green">
        <span class="dot"></span>
        <span id="signalText">GREEN LIGHT</span>
      </div>
      <div class="pill">
        <span style="color:var(--accent);">&#128663;</span>
        <span id="trackedText">0 Tracked</span>
      </div>
      <div class="pill">
        <span style="color:var(--danger);">&#9888;</span>
        <span id="violationCountPill">0 Violations</span>
      </div>
      <div class="pill">
        <span style="color:#a855f7;">&#9889;</span>
        <span id="fpsText">0.0 FPS</span>
      </div>
      <button id="soundToggle" class="sound-btn" title="Toggle audio siren">
        <svg id="soundIcon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M11 5L6 9H2v6h4l5 4V5z"/>
          <path d="M19.07 4.93a10 10 0 010 14.14M15.54 8.46a5 5 0 010 7.07"/>
        </svg>
      </button>
    </div>
  </header>

  <!-- Main View -->
  <main>
    <!-- Stream Column -->
    <section class="stream-section">
      <div class="video-card">
        <img id="liveFeed" src="/video_feed" alt="Live Traffic Stream" onerror="this.onerror=null; this.src=''; document.getElementById('offlineNotice').style.display='flex';" />
        
        <!-- Live HUD Overlays -->
        <div class="stream-overlay-top">
          <div class="overlay-badge">
            <span class="rec-indicator"></span>
            LIVE ENFORCEMENT FEED
          </div>
          <div id="streamResolution" class="overlay-badge">1080p | TensorRT GPU</div>
        </div>

        <div id="offlineNotice" style="display:none; position:absolute; inset:0; align-items:center; justify-content:center; flex-direction:column; gap:8px; color:var(--text-muted); font-size:13px; background:#080c14;">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M15 10l5 5m0-5l-5 5M4 19h16a2 2 0 002-2V7a2 2 0 00-2-2H4a2 2 0 00-2 2v10a2 2 0 002 2z"/>
          </svg>
          Connecting to StripeGuard Inference Engine...
        </div>
      </div>

      <!-- Quick Controls -->
      <div class="controls-bar">
        <span style="font-size:12px; color:var(--text-muted); font-weight:600;">SIGNAL OVERRIDE</span>
        <div class="btn-group">
          <button id="btnAuto" class="btn btn-active" onclick="setSignalMode('auto')">Auto Detect</button>
          <button id="btnRed" class="btn" onclick="setSignalMode('manual', 'RED')">Force RED</button>
          <button id="btnGreen" class="btn" onclick="setSignalMode('manual', 'GREEN')">Force GREEN</button>
        </div>
        <a href="/api/export_csv" download="violations.csv" class="btn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
          </svg>
          Export CSV
        </a>
      </div>
    </section>

    <!-- Right Column: Violations Feed -->
    <aside class="violation-section">
      <div class="panel-header">
        <div class="panel-title">
          <span>Violation Evidence Stream</span>
          <span id="violationCounterChip" class="count-chip">0</span>
        </div>
        <span style="font-size:11px; color:var(--text-muted);">Real-Time Photo Log</span>
      </div>

      <div id="violationFeed" class="feed-container">
        <div class="empty-state">No violations recorded yet.<br>Listening for zebra line incursions...</div>
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
        <div id="modalPlateImgContainer" style="display:none; margin-bottom:14px; background:rgba(0,0,0,0.5); padding:10px; border-radius:12px; border:1px solid var(--card-border); text-align:center;">
          <div style="font-size:11px; color:var(--text-muted); margin-bottom:6px; letter-spacing:0.5px; font-weight:700;">ENHANCED LICENSE PLATE CROP</div>
          <img id="modalPlateImg" src="" style="max-height:85px; max-width:100%; object-fit:contain; border-radius:6px;" alt="Enhanced Plate" />
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
          <h2 id="modalTitle" style="font-size:18px; font-weight:700;">Vehicle #ID-0</h2>
          <span id="modalTime" style="font-size:12px; color:var(--text-muted);">--:--:--</span>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; font-size:12px;">
          <div style="background:rgba(255,255,255,0.03); padding:10px; border-radius:10px; border:1px solid var(--card-border);">
            <div style="color:var(--text-muted); margin-bottom:4px;">License Plate (ANPR)</div>
            <div id="modalPlate" style="font-size:14px; font-weight:700; color:#38bdf8; font-family:monospace;">UNKNOWN</div>
          </div>
          <div style="background:rgba(255,255,255,0.03); padding:10px; border-radius:10px; border:1px solid var(--card-border);">
            <div style="color:var(--text-muted); margin-bottom:4px;">OCR Confidence</div>
            <div id="modalConfidence" style="font-size:14px; font-weight:700; color:#10b981;">0.0%</div>
          </div>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn" onclick="document.getElementById('evidenceModal').style.display='none'">Close</button>
        <a id="modalDownload" href="#" download class="btn btn-active">Download Evidence JPG</a>
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
        gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
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

    async function fetchStatus() {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();

        // Update Signal Pill
        const pill = document.getElementById('signalPill');
        const text = document.getElementById('signalText');
        pill.className = 'pill pill-signal ' + (data.signal === 'RED' ? 'signal-red' : (data.signal === 'YELLOW' ? 'signal-yellow' : 'signal-green'));
        text.innerText = data.signal + ' LIGHT';

        // Update counts
        document.getElementById('trackedText').innerText = data.tracked_vehicles + ' Tracked';
        document.getElementById('violationCountPill').innerText = data.violation_count + ' Violations';
        document.getElementById('fpsText').innerText = data.fps.toFixed(1) + ' FPS';

        // Update Controls Highlighting
        const isAuto = data.signal_mode === 'auto';
        document.getElementById('btnAuto').className = 'btn ' + (isAuto ? 'btn-active' : '');
        document.getElementById('btnRed').className = 'btn ' + (!isAuto && data.manual_signal === 'RED' ? 'btn-danger-active' : '');
        document.getElementById('btnGreen').className = 'btn ' + (!isAuto && data.manual_signal === 'GREEN' ? 'btn-success-active' : '');

      } catch (err) {
        console.warn('Status poll error', err);
      }
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
          feed.innerHTML = '<div class="empty-state">No violations recorded yet.<br>Listening for zebra line incursions...</div>';
          return;
        }

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
                <img src="/${cleanPath}" alt="ID ${id}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 24 24%22 fill=%22%23334155%22><text x=%2250%%22 y=%2250%%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 fill=%22white%22 font-size=%2210%22>PHOTO</text></svg>'"/>
              </div>
              <div class="violation-info">
                <div class="info-top">
                  <span class="vehicle-tag">Vehicle #${id}</span>
                  <span class="time-stamp">${time}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <span class="plate-badge">${plate}</span>
                  <span style="font-size:11px; color:var(--danger); font-weight:700;">RED ENCROACH</span>
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

    async function setSignalMode(mode, signal = 'RED') {
      try {
        await fetch('/api/signal', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode, signal })
        });
        fetchStatus();
      } catch (e) {
        console.error('Failed to set signal mode', e);
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
    """HTTP Request handler serving the minimalist web dashboard and live stream."""

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

            try:
                while True:
                    with STATE.lock:
                        frame_bytes = STATE.latest_frame_jpeg

                    if frame_bytes is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode("utf-8"))
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.033)  # ~30 FPS streaming
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        # 3. Status API
        elif path == "/api/status":
            summary = STATE.get_status_summary()
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
            with STATE.lock:
                violations = list(STATE.recent_violations)
            payload = json.dumps(violations).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
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
                self.send_header("Cache-Control", "public, max-age=86400")
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
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                mode = data.get("mode", "auto")
                signal = data.get("signal", "RED")
                STATE.set_signal_override(mode, signal)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except Exception as e:
                self.send_error(HTTPStatus.BAD_REQUEST, str(e))
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
        self.server = ThreadingHTTPServer((self.host, self.port), DashboardHTTPHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"\n=======================================================")
        print(f"  >>> StripeGuard AI Minimalist Frontend Running! <<<")
        print(f"  Open your browser: http://localhost:{self.port}")
        print(f"=======================================================\n")

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
    frame: np.ndarray | None,
    signal: str,
    tracked_count: int,
    violation_count: int,
    fps: float,
    source_fps: float = 0.0,
) -> None:
    STATE.update_frame(frame, signal, tracked_count, violation_count, fps, source_fps)

def add_violation_to_dashboard(entry: dict[str, Any]) -> None:
    STATE.add_violation(entry)

def get_signal_override() -> tuple[str, str]:
    return STATE.get_signal_override()


if __name__ == "__main__":
    # Test standalone execution
    server = start_dashboard(port=5000)
    print("Dashboard started in standalone mode with existing violation database. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        print("\nDashboard stopped.")
