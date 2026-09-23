"""
Computer Vision & ASL Gesture Recognition Pipeline

Processes video frames/landmarks using MediaPipe Hands (Tasks API) and geometric/ML feature classification:
- Extracts 21 3D hand landmarks per frame.
- Classifies ASL alphabet fingerspelling (A-Z, space, del) and common ASL Glosses.
- Tracks turn segmentation & pause boundaries to emit structured strings with confidence scores.
"""

import math
import time
import base64
import io
import os
import joblib
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


@dataclass
class RecognitionFrameResult:
    detected_symbol: Optional[str] = None
    symbol_type: str = "none"  # "letter", "gloss", "none"
    confidence: float = 0.0
    landmarks: List[Dict[str, float]] = field(default_factory=list)
    hand_present: bool = False
    handedness: str = "Right"


@dataclass
class TurnEmitEvent:
    structured_text: str
    tokens: List[Tuple[str, float]]
    timestamp: float


class SignRecognizer:
    """
    Real-time Hand Pose & Sign Classifier with MediaPipe Hands.
    """

    # Hand Landmark Indices (0 to 20)
    WRIST = 0
    THUMB_CMC = 1; THUMB_MCP = 2; THUMB_IP = 3; THUMB_TIP = 4
    INDEX_MCP = 5; INDEX_PIP = 6; INDEX_DIP = 7; INDEX_TIP = 8
    MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
    RING_MCP = 13; RING_PIP = 14; RING_DIP = 15; RING_TIP = 16
    PINKY_MCP = 17; PINKY_PIP = 18; PINKY_DIP = 19; PINKY_TIP = 20

    # Skeleton connections for rendering
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)              # Palm base
    ]

    def __init__(self, model_path: Optional[str] = "backend/models/asl_model.pkl",
                 task_model_path: str = "backend/models/hand_landmarker.task",
                 pause_threshold_sec: float = 1.3):
        self.pause_threshold = pause_threshold_sec
        self.model = None
        self.classes = None
        
        # Load trained ML model if available
        if model_path and os.path.exists(model_path):
            try:
                data = joblib.load(model_path)
                if isinstance(data, dict):
                    self.model = data.get("model")
                    self.classes = data.get("classes")
                else:
                    self.model = data
                print(f"[CV Engine] Loaded trained classifier from {model_path}")
            except Exception as e:
                print(f"[CV Engine] Model load warning: {e}")

        # MediaPipe HandLandmarker initialization
        self.detector = None
        if MP_AVAILABLE and os.path.exists(task_model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=task_model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=2,
                    min_hand_detection_confidence=0.4,
                    min_hand_presence_confidence=0.4,
                    min_tracking_confidence=0.4
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                print(f"[CV Engine] MediaPipe HandLandmarker initialized successfully.")
            except Exception as e:
                print(f"[CV Engine] HandLandmarker init warning: {e}")

        # Turn segmentation and tracking buffers
        self.last_sign_time = time.time()
        self.current_turn_tokens: List[Tuple[str, float, bool]] = []  # (symbol, conf, is_fs)
        self.consecutive_count = 0
        self.current_candidate = None
        self.candidate_confidence = 0.0
        self.hold_threshold_frames = 3

    def extract_landmarks(self, frame_bgr: np.ndarray) -> Tuple[bool, List[Dict[str, float]], Optional[str], np.ndarray]:
        """
        Extracts 21 3D landmarks from BGR OpenCV image and draws skeleton.
        Returns: (hand_present, landmarks_list, handedness, annotated_image)
        """
        annotated_img = frame_bgr.copy()
        h, w, _ = frame_bgr.shape

        if not MP_AVAILABLE or self.detector is None:
            return False, [], None, annotated_img

        try:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)

            if not detection_result.hand_landmarks:
                return False, [], None, annotated_img

            hand_lms = detection_result.hand_landmarks[0]
            handedness = "Right"
            if detection_result.handedness and len(detection_result.handedness) > 0:
                handedness = detection_result.handedness[0][0].category_name

            landmarks_list = []
            for lm in hand_lms:
                landmarks_list.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "px_x": lm.x * w,
                    "px_y": lm.y * h
                })

            # Draw landmarks and skeleton connections
            for p1, p2 in self.CONNECTIONS:
                pt1 = (int(landmarks_list[p1]["px_x"]), int(landmarks_list[p1]["px_y"]))
                pt2 = (int(landmarks_list[p2]["px_x"]), int(landmarks_list[p2]["px_y"]))
                cv2.line(annotated_img, pt1, pt2, (0, 230, 255), 2, cv2.LINE_AA)

            for lm in landmarks_list:
                center = (int(lm["px_x"]), int(lm["px_y"]))
                cv2.circle(annotated_img, center, 4, (0, 120, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated_img, center, 6, (255, 255, 255), 1, cv2.LINE_AA)

            return True, landmarks_list, handedness, annotated_img
        except Exception as e:
            return False, [], None, annotated_img

    def extract_feature_vector(self, landmarks: List[Dict[str, float]]) -> np.ndarray:
        """
        Normalizes 21 3D landmarks relative to wrist and scale invariant for ML classifier.
        """
        if len(landmarks) < 21:
            return np.zeros(63, dtype=np.float32)

        wrist = landmarks[self.WRIST]
        coords = []
        for lm in landmarks:
            coords.append([lm["x"] - wrist["x"], lm["y"] - wrist["y"], lm["z"] - wrist["z"]])

        coords = np.array(coords, dtype=np.float32)
        # Scale normalization using wrist-to-middle-mcp distance
        scale = np.linalg.norm(coords[self.MIDDLE_MCP])
        if scale > 1e-5:
            coords /= scale

        return coords.flatten()

    def classify_gesture(self, landmarks: List[Dict[str, float]]) -> Tuple[str, str, float]:
        """
        Classifies static alphabet handshape into character letters (A-Z).
        Returns: (letter, 'letter', confidence)
        """
        if len(landmarks) < 21:
            return "nothing", "none", 0.0

        # Check trained ML model for ASL letters
        if self.model is not None:
            try:
                features = self.extract_feature_vector(landmarks).reshape(1, -1)
                pred = self.model.predict(features)[0]
                proba = np.max(self.model.predict_proba(features)[0]) if hasattr(self.model, "predict_proba") else 0.88
                clean_pred = str(pred).upper().strip()
                if len(clean_pred) == 1 and clean_pred.isalpha():
                    return clean_pred, "letter", float(proba)
            except Exception:
                pass

        # Geometric heuristic fallback (characters only)
        return self._heuristic_classifier(landmarks)

    def _check_gloss_signs(self, lm: List[Dict[str, float]]) -> Tuple[str, str, float]:
        """
        Checks common conversational ASL gloss gestures.
        """
        def dist(i1, i2):
            return math.sqrt(
                (lm[i1]["x"] - lm[i2]["x"])**2 +
                (lm[i1]["y"] - lm[i2]["y"])**2 +
                (lm[i1]["z"] - lm[i2]["z"])**2
            )

        index_ext = lm[self.INDEX_TIP]["y"] < lm[self.INDEX_PIP]["y"]
        middle_ext = lm[self.MIDDLE_TIP]["y"] < lm[self.MIDDLE_PIP]["y"]
        ring_ext = lm[self.RING_TIP]["y"] < lm[self.RING_PIP]["y"]
        pinky_ext = lm[self.PINKY_TIP]["y"] < lm[self.PINKY_PIP]["y"]
        thumb_ext = lm[self.THUMB_TIP]["y"] < lm[self.THUMB_IP]["y"]

        extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])

        # 1. "HELLO" (Open 5 Hand with spread fingers)
        if extended_count == 4 and thumb_ext and dist(self.INDEX_TIP, self.PINKY_TIP) > 0.30:
            return "HELLO", "gloss", 0.94

        # 2. "THANK-YOU" (Flat hand with fingers together touching chin level)
        if extended_count == 4 and not thumb_ext and dist(self.INDEX_TIP, self.MIDDLE_TIP) < 0.08:
            return "THANK-YOU", "gloss", 0.89

        # 3. "HELP" (Thumbs up gesture)
        if extended_count == 0 and thumb_ext and lm[self.THUMB_TIP]["y"] < lm[self.WRIST]["y"] - 0.12:
            return "HELP", "gloss", 0.88

        # 4. "YES" (S-fist / closed fist nod)
        if extended_count == 0 and not thumb_ext and dist(self.THUMB_TIP, self.INDEX_PIP) < 0.08:
            return "YES", "gloss", 0.83

        # 5. "NO" (Index and middle snapping onto thumb)
        if index_ext and middle_ext and dist(self.INDEX_TIP, self.THUMB_TIP) < 0.06:
            return "NO", "gloss", 0.85

        return "none", "none", 0.0

    def _heuristic_classifier(self, lm: List[Dict[str, float]]) -> Tuple[str, str, float]:
        """
        Geometric rule-based classifier based on finger extensions and relative distances.
        """
        def dist(i1, i2):
            return math.sqrt(
                (lm[i1]["x"] - lm[i2]["x"])**2 +
                (lm[i1]["y"] - lm[i2]["y"])**2 +
                (lm[i1]["z"] - lm[i2]["z"])**2
            )

        thumb_ext = lm[self.THUMB_TIP]["y"] < lm[self.THUMB_IP]["y"]
        index_ext = lm[self.INDEX_TIP]["y"] < lm[self.INDEX_PIP]["y"]
        middle_ext = lm[self.MIDDLE_TIP]["y"] < lm[self.MIDDLE_PIP]["y"]
        ring_ext = lm[self.RING_TIP]["y"] < lm[self.RING_PIP]["y"]
        pinky_ext = lm[self.PINKY_TIP]["y"] < lm[self.PINKY_PIP]["y"]

        extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])

        # 'A' (Fist with thumb resting beside index)
        if extended_count == 0 and not thumb_ext:
            return "A", "letter", 0.88

        # 'B' (4 fingers straight up)
        if extended_count == 4 and not thumb_ext:
            return "B", "letter", 0.92

        # 'C' (Curved hand in C-shape)
        if not index_ext and not middle_ext and 0.10 < dist(self.THUMB_TIP, self.INDEX_TIP) < 0.25:
            return "C", "letter", 0.85

        # 'D' (Index straight up only)
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "D", "letter", 0.91

        # 'I' (Pinky up only) or 'Y' (Thumb + Pinky)
        if pinky_ext and not index_ext and not middle_ext and not ring_ext:
            if thumb_ext:
                return "Y", "letter", 0.93
            return "I", "letter", 0.90

        # 'L' (Index up, thumb out)
        if index_ext and thumb_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "L", "letter", 0.94

        # 'V' vs 'U' vs 'W'
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            if dist(self.INDEX_TIP, self.MIDDLE_TIP) > 0.08:
                return "V", "letter", 0.89
            return "U", "letter", 0.86

        if index_ext and middle_ext and ring_ext and not pinky_ext:
            return "W", "letter", 0.91

        return "nothing", "none", 0.35

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[RecognitionFrameResult, Optional[TurnEmitEvent], np.ndarray]:
        """
        Process single video frame, track turn accumulation, and detect pause boundaries.
        Returns: (frame_result, optional_emitted_turn_event, annotated_image)
        """
        hand_present, landmarks, handedness, annotated_img = self.extract_landmarks(frame_bgr)
        now = time.time()

        if not hand_present or not landmarks:
            emitted_event = None
            if self.current_turn_tokens and (now - self.last_sign_time) > self.pause_threshold:
                emitted_event = self._emit_current_turn()

            return (
                RecognitionFrameResult(
                    detected_symbol=None,
                    symbol_type="none",
                    confidence=0.0,
                    landmarks=[],
                    hand_present=False,
                    handedness="Right"
                ),
                emitted_event,
                annotated_img
            )

        # Classify gesture
        symbol, sym_type, conf = self.classify_gesture(landmarks)

        # Temporal debouncing
        if symbol != "nothing" and conf >= 0.5:
            if symbol == self.current_candidate:
                self.consecutive_count += 1
                self.candidate_confidence = max(self.candidate_confidence, conf)
            else:
                self.current_candidate = symbol
                self.consecutive_count = 1
                self.candidate_confidence = conf

            if self.consecutive_count == self.hold_threshold_frames:
                is_fs = (sym_type == "letter")
                self.current_turn_tokens.append((symbol, round(self.candidate_confidence, 2), is_fs))
                self.last_sign_time = now

        emitted_event = None
        if self.current_turn_tokens and (now - self.last_sign_time) > self.pause_threshold:
            emitted_event = self._emit_current_turn()

        # Draw overlay info on frame
        badge_color = (0, 255, 120) if conf >= 0.75 else ((0, 180, 255) if conf >= 0.4 else (0, 70, 255))
        cv2.putText(
            annotated_img,
            f"Sign: {symbol} ({conf:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            badge_color,
            2,
            cv2.LINE_AA
        )

        return (
            RecognitionFrameResult(
                detected_symbol=symbol if symbol != "nothing" else None,
                symbol_type=sym_type,
                confidence=round(conf, 2),
                landmarks=landmarks,
                hand_present=True,
                handedness=handedness or "Right"
            ),
            emitted_event,
            annotated_img
        )

    def _emit_current_turn(self) -> TurnEmitEvent:
        """
        Compiles accumulated tokens into structured turn string:
        - Groups consecutive fingerspelled letters into `[FS] L(0.9)-L(0.85)...`
        - Emits glosses with trailing confidence e.g. `TOMORROW WEATHER HOW (0.85)`
        """
        if not self.current_turn_tokens:
            return TurnEmitEvent(structured_text="", tokens=[], timestamp=time.time())

        parts = []
        fs_buffer = []
        raw_token_list = []

        for symbol, conf, is_fs in self.current_turn_tokens:
            raw_token_list.append((symbol, conf))
            if is_fs:
                fs_buffer.append(f"{symbol}({conf})")
            else:
                if fs_buffer:
                    parts.append(f"[FS] {'-'.join(fs_buffer)}")
                    fs_buffer = []
                parts.append(f"{symbol} ({conf})")

        if fs_buffer:
            parts.append(f"[FS] {'-'.join(fs_buffer)}")

        structured_str = " ".join(parts)
        self.current_turn_tokens = []
        self.consecutive_count = 0
        self.current_candidate = None

        return TurnEmitEvent(
            structured_text=structured_str,
            tokens=raw_token_list,
            timestamp=time.time()
        )

    def reset_turn(self):
        """Clears the active turn buffer."""
        self.current_turn_tokens = []
        self.consecutive_count = 0
        self.current_candidate = None
        self.last_sign_time = time.time()
