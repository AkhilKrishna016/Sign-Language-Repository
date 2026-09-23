# Sign Language & Neural Gesture Intelligence Repository

[![GitHub](https://img.shields.io/badge/GitHub-Sign--Language--Repository-181717.svg?style=flat&logo=github)](https://github.com/AkhilKrishna016/Sign-Language-Repository)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB.svg?style=flat&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Solutions-007ACC.svg?style=flat)](https://developers.google.com/mediapipe)
[![TensorFlow Lite](https://img.shields.io/badge/TFLite-MobileNet-FF6F00.svg?style=flat&logo=TensorFlow)](https://www.tensorflow.org/lite)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=PyTorch)](https://pytorch.org)

A comprehensive monorepo consolidating two complete, production-ready sign language communication and neural gesture intelligence projects.

---

## Repository Projects

This repository hosts two distinct, standalone systems:

```text
Sign-Language-Repository/
│
├── 📂 EchoSign-Bidirectional-Communication/
│   └── Real-time bidirectional communication bridge between mute and deaf individuals
│       (PyTorch MobileNetV2 + MLP, MediaPipe Landmarker, FastAPI, WebSockets, STT/TTS, Streamlit)
│
└── 📂 OmniSign-Object-Gesture-Detection/
    └── Real-time neural sign gesture and custom object detection engine
        (TFLite MobileNet SSD, MediaPipe Hands, TFLite Classifier, FastAPI Server, Modern Dark Web UI, Android Java App)
```

---

## Project 1: EchoSign &mdash; Bidirectional Communication System

> **Directory**: [`EchoSign-Bidirectional-Communication/`](./EchoSign-Bidirectional-Communication/)  
> **Documentation**: [EchoSign README](./EchoSign-Bidirectional-Communication/README.md)

### Highlights:
- **Deaf $\to$ Mute (Sign to Voice/Text)**:
  - High-precision dual-pipeline computer vision engine:
    - **Fingerspelling (A–Z, 0–9)**: 63-dim normalized hand landmark MLP classifier (99.8% test accuracy) + 224x224 RGB MobileNetV2 feature extractor.
    - **Word-Level Gestures**: Landmark trajectory temporal feature classification.
  - Natural audio synthesis using `gTTS` and `pyttsx3` with caching.
- **Mute $\to$ Deaf (Speech/Text to Animated Sign Visuals)**:
  - Real-time speech transcription (Google Speech Recognition / Web Speech API).
  - NLP lemmatization, word tokenization, and fingerspelling fallback.
  - Interactive avatar rendering sign videos, animated GIFs, and fingerspelling sequences.
- **Frontend Interfaces**:
  - Interactive **Streamlit Dashboard** (`streamlit_app.py`).
  - Ultra-low-latency **FastAPI WebSocket Web Interface** (`frontend/`).

### Quick Start:
```bash
cd EchoSign-Bidirectional-Communication
pip install -r requirements.txt
python run.py
```

---

## Project 2: OmniSign &mdash; Neural Object & Gesture Detection

> **Directory**: [`OmniSign-Object-Gesture-Detection/`](./OmniSign-Object-Gesture-Detection/)  
> **Documentation**: [OmniSign README](./OmniSign-Object-Gesture-Detection/README.md)

### Highlights:
- **Dual-Stage Hybrid Neural Pipeline**:
  - **MediaPipe Hands**: Real-time 21 3D hand keypoints localization with confidence filtering.
  - **SSD MobileNet TFLite**: Multi-class object localization for bounding boxes and labels.
  - **TFLite Hand Gesture Classifier**: Fast edge inference classifying signs (`LikeYou`, `ThankYou`, `Yes`).
- **High-Performance FastAPI Server**:
  - Simultaneous support for HTTP multipart frame uploads (`/detect`) and full-duplex WebSocket live streams (`/ws/detect`).
  - Non-blocking asynchronous processing yielding 30+ FPS.
- **Futuristic Glassmorphism Web Interface**:
  - Live canvas rendering with dynamic bounding boxes, skeleton wireframes, and confidence telemetry HUD.
  - Tunable detection sensitivity, smoothing window, and label filtering.
- **Multi-Platform Source**:
  - Native Android Java application (`CustomObjectDetectionLiveFeedJava-main`).
  - Google Colab / Jupyter model training notebook (`Object_Detection.ipynb`) and Roboflow Pascal VOC datasets.
  - Academic course deliverables, presentations (`.pptx`), and project documentation (`OPPS/`).

### Quick Start:
```bash
cd OmniSign-Object-Gesture-Detection
pip install -r requirements.txt
python run_server.py
```

---

## Technology Stack Matrix

| Technology | EchoSign | OmniSign |
| :--- | :---: | :---: |
| **FastAPI + Uvicorn** | :white_check_mark: | :white_check_mark: |
| **Full-Duplex WebSockets** | :white_check_mark: | :white_check_mark: |
| **MediaPipe Hands** | :white_check_mark: | :white_check_mark: |
| **TensorFlow Lite (TFLite)** | &mdash; | :white_check_mark: |
| **PyTorch (MobileNetV2)** | :white_check_mark: | &mdash; |
| **Speech Recognition & TTS** | :white_check_mark: | &mdash; |
| **Streamlit Interface** | :white_check_mark: | &mdash; |
| **Modern Dark-Mode Web UI** | :white_check_mark: | :white_check_mark: |
| **Native Android Java Client** | &mdash; | :white_check_mark: |
| **Model Training Pipelines** | :white_check_mark: | :white_check_mark: |

---

## Repository Structure Overview

```text
.
├── README.md                                    # This master overview
├── .gitignore                                   # Global repository exclusion rules
│
├── EchoSign-Bidirectional-Communication/        # Project 1: Bidirectional System
│   ├── backend/                                 # CV, NLP, TTS, STT, and FastAPI routers
│   ├── data/                                    # Landmark feature datasets
│   ├── dataset/                                 # Character and gesture image assets
│   ├── frontend/                                # Modern dark-mode web client
│   ├── logs/                                    # System telemetry & feedback logs
│   ├── scripts/                                 # Data collection & model training utilities
│   ├── tests/                                   # Unit and integration test suite
│   ├── requirements.txt                         # EchoSign dependencies
│   ├── run.py                                   # EchoSign unified launcher
│   ├── streamlit_app.py                         # Streamlit UI
│   └── README.md                                # EchoSign complete documentation
│
└── OmniSign-Object-Gesture-Detection/           # Project 2: OmniSign Live System
    ├── server.py                                # FastAPI + TFLite inference server
    ├── run_server.py                            # OmniSign runner with browser auto-open
    ├── requirements.txt                         # OmniSign dependencies
    ├── web/                                     # Glassmorphism HTML5/CSS3/ES6 Web UI
    ├── OOP cp/                                  # TFLite models, Android Java project, VOC datasets
    │   ├── model.tflite
    │   ├── converted_tflite (4)/
    │   ├── ModelTrainingCodeFile/
    │   └── CustomObjectDetectionLiveFeedJava-main/
    ├── OPPS/                                    # Presentations (.pptx), reports & synopsis
    └── README.md                                # OmniSign complete documentation
```

---

## License

This repository is maintained for academic research, assistive technology, and machine learning development.
