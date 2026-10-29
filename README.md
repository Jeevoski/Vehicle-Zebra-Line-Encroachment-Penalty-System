# StripeGuard AI

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/Detector-YOLOv8-111827)](https://github.com/ultralytics/ultralytics)

StripeGuard AI is an end-to-end intelligent transportation system prototype for detecting vehicles that enter a zebra crossing while a traffic signal is red. It tracks vehicles, extracts plate text, preserves image evidence, and writes structured JSON and CSV violation records.

## Features

- HSV dual-range red-light detection over a configurable traffic-light ROI.
- Manual signal simulation with `r` and `g`, plus `a` to return to automatic mode.
- YOLOv8 vehicle detection with persistent ByteTrack IDs.
- Bottom-center point-in-polygon zebra-crossing incursion logic.
- One penalty per tracked vehicle per red-light cycle.
- CLAHE and bilateral-filter preprocessing with EasyOCR ANPR.
- Evidence images in `violations/` and machine-readable logs.
- Interactive ROI selection utility for new camera geometry.

## Architecture

```text
Video source
    |
    +--> HSV signal detector ----> RED/GREEN state
    |
    +--> YOLOv8 + ByteTrack ------> vehicle boxes + persistent IDs
                                      |
                       bottom-center point in zebra polygon
                                      |
                         RED + inside + new ID = violation
                                      |
                    crop -> CLAHE/OCR -> evidence + JSON/CSV
```

## Installation

Prerequisites: Python 3.10+, a camera/video file, and enough RAM for PyTorch and EasyOCR. A CUDA-capable NVIDIA GPU is recommended for deployment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run downloads the configured YOLO weights and EasyOCR language model. Put a sample video at `data/traffic_sample.mp4`, or pass another source with `--source`.

## Usage

1. Copy the sample video into `data/traffic_sample.mp4`.
2. Run the ROI tool and copy its printed snippets into `config.py`:

   ```powershell
   python roi_selector.py
   ```

   Select four zebra-corner points, press Enter, select two signal-ROI corners, then press Enter.

   For ETS2, set the game camera first and select from the monitor instead:

   ```powershell
   python roi_selector.py --screen
   ```

   Copy the printed `ZEBRA_ROI` and `SIGNAL_ROIS` snippets into `config.py` before starting screen monitoring.

3. Start monitoring:

   ```powershell
   python main.py
   ```

   Useful options include `--source 0` for a webcam, `--signal-mode manual` for simulation, `--headless` for servers or non-GUI environments, `--output-video violations/annotated_detection.mp4` to save the detection overlay, and `--cpu-ocr` when no GPU is available. During execution, `r` forces RED, `g` forces GREEN, `a` restores automatic signal detection, and `q` exits.

   For offline replay with an annotated output video:

   ```powershell
   python main.py --source data/traffic_sample.mp4 --headless --cpu-ocr --output-video violations/annotated_detection.mp4
   ```

   To ingest the configured Fresno, California YouTube traffic camera directly with a low-latency latest-frame reader:

   ```powershell
   python main.py --live
   ```

   The equivalent explicit command is `python main.py --source https://www.youtube.com/live/sTF-6_xinUU`. Streamlink resolves the best available `1080p`, `720p`, or fallback stream, while a background reader drops stale frames so inference stays near live time. Frames are processed and displayed at their native resolution without aspect-ratio scaling. Use `--headless` for a server and `--output-video violations/live_annotated.mp4` when recording annotated output.

   To use Euro Truck Simulator 2 as the camera source, set the in-game camera so the traffic signal, vehicles, and zebra crossing are visible, then capture the primary monitor:

   ```powershell
   pip install mss
   python main.py --screen --signal-mode auto --output-video violations/ets2_annotated.mp4
   ```

   To capture only the game window area, use `--screen-region x,y,width,height`, for example `--screen-region 0,0,1920,1080`. Copy the ROI values from `roi_selector.py` into `config.py` for the captured ETS2 view. Press `q` in the OpenCV window to stop. Screen capture analyses pixels only; it does not read or modify ETS2 game files.

## CARLA simulation

For a small local test that does not require CARLA, use the OpenCV-only synthetic simulator:

```powershell
python synthetic_simulator.py --duration 30
```

This creates `data/synthetic_traffic.mp4` and `data/synthetic_rois.json` with a road, zebra crossing, traffic light, moving vehicle shapes, and alternating red/green phases. Copy the ROI values from the JSON file into `config.py`, then replay it through the detector:

```powershell
python main.py --source data/synthetic_traffic.mp4 --headless --cpu-ocr --output-video violations/synthetic_annotated.mp4
```

This lightweight scene is suitable for testing ROI scaling, signal detection, video writing, and the end-to-end workflow. Its drawn vehicle shapes are not a substitute for real training data, so use real traffic footage or a labeled dataset to evaluate YOLO accuracy.

The live-camera ROIs are specific to one camera view. For repeatable testing, use the optional CARLA scenario runner. Install and start a CARLA release that matches the Python API available in your environment, then run:

```powershell
python carla_simulator.py --duration 60 --output data/carla_traffic.mp4 --roi-output data/carla_rois.json
```

The runner spawns autopilot traffic, records a fixed camera, alternates red and green phases, and overlays a visible zebra crossing and signal panel. It also writes the exact `1280x720` ROIs used by the recording. Replay the result through StripeGuard:

```powershell
python main.py --source data/carla_traffic.mp4 --headless --cpu-ocr --output-video violations/carla_annotated.mp4
```

For a first detector check, copy the `zebra_roi` and `signal_rois` values from `data/carla_rois.json` into `config.py`. CARLA is intentionally optional and is not included in `requirements.txt` because its Python package must match the installed CARLA server version.

4. Inspect evidence and logs in `violations/`.

Each image uses `ID{track_id}_{plate_number}_{YYYYMMDD_HHMMSS}.jpg`. The JSON and CSV records contain timestamp, vehicle ID, plate text, OCR confidence, and evidence path.

## Configuration

Edit `config.py` for camera geometry, model path, confidence thresholds, OCR behavior, and HSV ranges. The checked-in ROI is calibrated for the supplied 1920x1080 footage and should be recalibrated for each camera angle.

## Edge deployment

For NVIDIA Jetson, install the JetPack-compatible PyTorch and Torchvision wheels before the remaining requirements. Export a TensorRT engine for YOLO where supported, reduce input resolution to match the camera, and use a hardware-accelerated GStreamer source. Keep OCR asynchronous or sample it only on a newly penalized track to preserve real-time throughput. Use a UPS, local evidence retention policy, synchronized system time, and encrypted log transfer for a production enforcement deployment.

## Operational considerations

This repository provides the vision and evidence pipeline. A production penalty system also needs camera calibration, plate-image quality validation, privacy and retention controls, operator audit trails, legal review, and a secure downstream citation service.
