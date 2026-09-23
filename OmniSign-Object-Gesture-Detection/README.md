# OmniSign Live &bull; Real-Time Neural Object & Sign Gesture Detection

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands%20Tracking-007ACC.svg?style=flat)](https://developers.google.com/mediapipe)
[![TensorFlow Lite](https://img.shields.io/badge/TFLite-MobileNet%20SSD-FF6F00.svg?style=flat&logo=TensorFlow)](https://www.tensorflow.org/lite)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB.svg?style=flat&logo=python)](https://python.org)

OmniSign is an edge-optimized, real-time sign language recognition and object detection system. It employs a **dual-stage hybrid inference pipeline** combining **MediaPipe Hands** for anatomical landmark tracking, **SSD MobileNet TFLite** for multi-class object localization, and a fine-tuned **TFLite classification network** for hand gesture interpretation.

---

## Key Features

- **Dual-Stage Neural Inference**:
  - **MediaPipe Hand Tracking**: Dynamically tracks 21 3D hand landmarks in real time with high spatial accuracy.
  - **SSD MobileNet Object Detector**: Localizes detected classes and bounding boxes with confidence scores.
  - **TFLite Hand Gesture Classifier**: Fast, low-latency gesture recognition (`LikeYou`, `ThankYou`, `Yes`, etc.).
- **High-Performance FastAPI Backend**:
  - Dual ingestion modes: HTTP multipart image uploads (`/detect`) and low-latency duplex WebSocket streams (`/ws/detect`).
  - Asynchronous non-blocking architecture delivering 30+ FPS processing.
- **Futuristic Dark-Mode Web Dashboard**:
  - Live webcam canvas feed with bounding boxes, keypoint overlays, and gesture telemetry.
  - Interactive parameter controls: confidence thresholds, smoothing, and active class selection.
  - Real-time FPS and round-trip latency HUD meters.
- **Multi-Platform Ecosystem**:
  - Python/FastAPI web server.
  - Native Android Java application source (`CustomObjectDetectionLiveFeedJava-main`).
  - Google Colab / Jupyter training pipeline (`Object_Detection.ipynb`) with Roboflow Pascal VOC datasets.
  - Academic reports, presentations, and design synopses (`OPPS/`).

---

## Architecture Pipeline

```text
[Webcam / Client Stream]
          │
          ▼ (WebSocket / Base64 Frames)
[FastAPI Server (server.py)]
          │
   ┌──────┴──────────────────────────┐
   ▼                                 ▼
[MediaPipe Hands Tracker]   [SSD MobileNet Detector]
(21 Hand Keypoints)          (Bounding Boxes & Classes)
   │                                 │
   └──────────────┬──────────────────┘
                  ▼
       [Crop & Normalization]
                  ▼
   [TFLite Gesture Classifier]
       (LikeYou / ThankYou / Yes)
                  ▼
 [JSON Telemetry + Annotated Frame]
                  │
                  ▼
[Interactive Web Dashboard (HTML5/CSS3/ES6)]
```

---

## Directory Structure

```text
OmniSign-Object-Gesture-Detection/
├── server.py                               # FastAPI backend with hybrid neural inference
├── run_server.py                           # Launcher script with auto-browser opening
├── requirements.txt                        # Dependency list
├── README.md                               # Project documentation
├── web/                                    # Frontend Web Interface
│   ├── index.html                          # Dashboard layout & controls
│   ├── styles.css                          # Modern glassmorphism UI styles
│   └── app.js                              # WebSocket streaming & canvas rendering
├── OOP cp/                                 # Models, Datasets & Android Source
│   ├── model.tflite                        # Object detector TFLite weights
│   ├── converted_tflite (4)/               # Gesture classifier weights & labels
│   ├── ModelTrainingCodeFile/              # Object_Detection.ipynb
│   ├── CustomObjectDetectionLiveFeedJava-main/ # Android Java mobile app source
│   ├── oops_cp/                            # Cropped gesture training images
│   ├── kjc s.v1i.voc/                      # Roboflow Pascal VOC dataset 1
│   └── njsnjv.v1i.voc/                     # Roboflow Pascal VOC dataset 2
└── OPPS/                                   # Presentations & Academic Reports
    ├── End-sem-ppt.pptx
    ├── mid-sem-ppt.pptx
    └── course reports & synopsis
```

---

## Quick Start Guide

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11
- Webcam connected to your workstation

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/AkhilKrishna016/Sign-Language-Repository.git
cd Sign-Language-Repository/OmniSign-Object-Gesture-Detection

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Server
Launch the server with the automated runner:
```bash
python run_server.py
```
This automatically boots the FastAPI server on port `8000` and launches your default web browser to:
- **Web Interface**: `http://localhost:8000`
- **Interactive API Documentation (Swagger)**: `http://localhost:8000/docs`

---

## Supported Classes & Gestures

| Category | Classes |
| :--- | :--- |
| **Active Object Detections** | `i like you`, `Thank You` |
| **Active Gesture Classifications** | `LikeYou`, `ThankYou`, `Yes` |
| **Hand Landmarks** | 21 3D Keypoints per hand (Wrist, Thumb, Index, Middle, Ring, Pinky) |
