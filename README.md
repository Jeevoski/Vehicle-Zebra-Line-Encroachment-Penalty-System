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

3. Start monitoring:

   ```powershell
   python main.py
   ```

   Useful options include `--source 0` for a webcam, `--signal-mode manual` for simulation, `--headless` for servers or non-GUI environments, `--output-video violations/annotated_detection.mp4` to save the detection overlay, and `--cpu-ocr` when no GPU is available. During execution, `r` forces RED, `g` forces GREEN, `a` restores automatic signal detection, and `q` exits.

   For offline replay with an annotated output video:

   ```powershell
   python main.py --source data/traffic_sample.mp4 --headless --cpu-ocr --output-video violations/annotated_detection.mp4
   ```

4. Inspect evidence and logs in `violations/`.

Each image uses `ID{track_id}_{plate_number}_{YYYYMMDD_HHMMSS}.jpg`. The JSON and CSV records contain timestamp, vehicle ID, plate text, OCR confidence, and evidence path.

## Configuration

Edit `config.py` for camera geometry, model path, confidence thresholds, OCR behavior, and HSV ranges. The checked-in ROI is calibrated for the supplied 1920x1080 footage and should be recalibrated for each camera angle.

## Edge deployment

For NVIDIA Jetson, install the JetPack-compatible PyTorch and Torchvision wheels before the remaining requirements. Export a TensorRT engine for YOLO where supported, reduce input resolution to match the camera, and use a hardware-accelerated GStreamer source. Keep OCR asynchronous or sample it only on a newly penalized track to preserve real-time throughput. Use a UPS, local evidence retention policy, synchronized system time, and encrypted log transfer for a production enforcement deployment.

## Operational considerations

This repository provides the vision and evidence pipeline. A production penalty system also needs camera calibration, plate-image quality validation, privacy and retention controls, operator audit trails, legal review, and a secure downstream citation service.
