# EchoSign — Bidirectional Sign-Language Communication Assistant

> **End-to-end AI assistant bridging Deaf/Mute signers and Hearing conversation partners using hand gesture recognition (ASL), natural language understanding, live captions, and speech synthesis.**

EchoSign recognizes both **full ASL signs (word-level glosses)** and **fingerspelled letters (the alphabet)** using MediaPipe 3D hand landmark extraction, reconstructs natural conversational English meaning using a dedicated language-understanding layer, holds a conversation, and closes the loop back with audio speech and visual ASL feedback.

---

## 🌟 Key Architecture & Build Order

```mermaid
graph TD
    subgraph 1. Hand Tracking (MediaPipe Hands)
        Cam[Webcam Video Stream] --> Tracker[Modular HandTracker]
        Tracker --> Landmarks[21 3D Landmarks per hand - up to 2 hands]
    end

    subgraph 2. Dual Classifiers (Landmark Coordinates Only - No Raw Pixels)
        Landmarks --> FS[Fingerspelling Classifier: MLP/RF on Landmarks]
        Landmarks --> Buffer[Rolling Window: 15-30 frames]
        Buffer --> WordClf[Word-Level Temporal Sign Classifier]
    end

    subgraph 3. Turn Segmentation & Utterance Buffering
        FS & WordClf --> Segmenter[Turn Segmenter: Hand at Rest Pause Detection]
        Segmenter -->|Structured Turn String| StructuredMsg[GLOSS confidence + [FS] A(0.9)-B(0.8)]
    end

    subgraph 4. Dialogue Layer & Feedback Loop
        StructuredMsg --> LangEngine[Dialogue Layer: sign-language-assistant-system-prompt.md]
        LangEngine --> Grammar[ASL Grammar Inversion + M/N, U/V Silent Correction]
        LangEngine --> FeedbackLog[Feedback Logger: logs/low_confidence_feedback.jsonl]
    end

    subgraph 5. Output & User Interfaces
        Grammar --> Captions[Live High-Contrast Subtitles]
        Grammar --> TTS[Text-to-Speech Voice Engine]
        Captions & TTS --> UI[FastAPI Web Interface / Streamlit App]
    end
```

---

## 📦 Deliverables & Features

### 1. Dedicated System Prompt Asset
- [`sign-language-assistant-system-prompt.md`](./sign-language-assistant-system-prompt.md): The official language-understanding prompt covering gloss parsing, topic-first ASL grammar, silent correction for visual confusion pairs ($M/N$, $U/V$, $K/P$, $A/S/T$), and 3-tier uncertainty gating ($\ge 0.75$ reliable, $0.4\text{–}0.75$ clarify, $< 0.4$ repeat).

### 2. Preprocessing Scripts (`scripts/`)
- [`scripts/preprocess_landmarks.py`](./scripts/preprocess_landmarks.py): Extracts 21 3D landmarks per hand from image datasets (ASL Alphabet / Sign Language MNIST) and saves normalized coordinate arrays to `data/asl_landmarks.npz` (never trains directly on raw pixels).
- [`scripts/preprocess_wlasl.py`](./scripts/preprocess_wlasl.py): Extracts rolling temporal landmark sequences from WLASL videos and saves to `data/wlasl_sequences.npz`.

### 3. Dual Classifier Training (`scripts/`)
- [`scripts/train_fingerspelling.py`](./scripts/train_fingerspelling.py): Trains lightweight MLP (**98.41% acc**) and Random Forest (**97.62% acc**) models on landmark coordinates, saved to `backend/models/`.
- [`scripts/train_word_level.py`](./scripts/train_word_level.py): Trains temporal sequence classifier over rolling landmark windows for dynamic ASL glosses (`HELLO`, `THANK-YOU`, `HELP`, `WHERE`, etc.), saved to `backend/models/word_level_model.pkl`.

### 4. Modular Pipeline (`backend/`)
- `HandTracker`: MediaPipe Tasks HandLandmarker extracting 21 3D points for up to 2 hands.
- `FingerspellingClassifier`: Alphabet classifier with per-letter confidence scoring.
- `WordLevelClassifier`: Temporal trajectory classifier over a 20-frame rolling window.
- `TurnSegmenter`: "Hand at rest" pause detection emitting structured strings.
- `FeedbackLogger`: Structured logging to `logs/low_confidence_feedback.jsonl` for continuous retraining.
- `SpeechToSignEngine`: Reverse channel translating English to ASL Gloss and visual cards.
- `TTSEngine`: Voice audio generation via gTTS and offline pyttsx3.

---

## 🚀 Quick Start

### 1. Launch Production FastAPI Web Interface
```bash
python run.py
```
Open **[http://localhost:8000](http://localhost:8000)**.

### 2. Launch Streamlit Alternative Demo
```bash
streamlit run streamlit_app.py
```

### 3. Run Automated Test Suite (23/23 tests passed)
```bash
python -m unittest discover tests
```

### 4. Preprocess Datasets & Retrain Models
```bash
# Step 1: Preprocess images and video sequences to landmarks
python scripts/preprocess_landmarks.py
python scripts/preprocess_wlasl.py

# Step 2: Train classifiers on landmarks
python scripts/train_fingerspelling.py
python scripts/train_word_level.py
```

---

## ⚠️ Known Limitations & Out-of-Scope (v1)

1. **Static vs. Motion Letters ($J$ and $Z$)**:
   - Letters $J$ and $Z$ involve hand motion trajectories. In single static frames, they may produce visual ambiguity; the dialogue layer is specifically designed to ask for a repeat if repeated garbling occurs on these letters.
2. **Lighting & Background Sensitivity**:
   - Because all classification operates strictly on 3D landmark coordinates (rather than raw pixel CNNs), background variations are naturally filtered. However, severe low-light or extreme glare can impact landmark detector confidence.
3. **Sign-Avatar / Text-to-Sign Avatar Synthesis**:
   - Rendering photorealistic 3D signing avatars from text is a separate, harder research frontier and is explicitly flagged as out-of-scope for v1. EchoSign provides live captions, speech audio, and visual ASL flashcard cues.
4. **Language Specificity**:
   - The vocabulary, grammar inversion rules, and datasets are tailored specifically for **American Sign Language (ASL)**. Other sign languages (BSL, ISL, Auslan) use different manual alphabets and grammatical structures.

---

## 📂 Project Structure

```
├── backend/
│   ├── classifiers/
│   │   ├── fingerspelling_classifier.py  # Single-frame landmark alphabet classifier
│   │   └── word_level_classifier.py      # Rolling-window temporal sign classifier
│   ├── hand_tracker.py                   # Modular MediaPipe 3D hand tracker
│   ├── turn_segmentation.py              # Hand-at-rest pause detector
│   ├── language_engine.py                # Prompt-compliant dialogue layer
│   ├── speech_to_sign.py                 # Reverse English -> ASL Gloss translator
│   ├── tts_engine.py                     # Text-to-Speech audio synthesizer
│   ├── feedback_logger.py                # Low-confidence feedback loop recorder
│   ├── main.py                           # FastAPI REST & WebSocket server
│   └── models/                           # Trained model weights & task assets
├── data/                                 # Preprocessed landmark NPZ arrays
├── dataset/                              # ASL Alphabet training & test dataset
├── frontend/                             # Modern dark-mode web application
├── logs/                                 # low_confidence_feedback.jsonl
├── scripts/
│   ├── preprocess_landmarks.py           # Images -> 3D landmarks
│   ├── preprocess_wlasl.py               # Videos -> Temporal sequences
│   ├── train_fingerspelling.py           # Train MLP & Random Forest
│   └── train_word_level.py               # Train temporal sequence model
├── tests/                                # Comprehensive test suite (23 tests)
├── sign-language-assistant-system-prompt.md # System prompt asset
├── streamlit_app.py                      # Streamlit application
├── run.py                                # One-click runner
└── README.md                             # Documentation
```
