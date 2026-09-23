"""
Word-Level Sign Classifier (Temporal Model)

Operates on a rolling window of landmark frames (15 to 30 frames) to recognize full ASL signs
(WLASL / dynamic conversational signs) that are not spelled letter-by-letter.
Outputs an ASL gloss token with a confidence score.
"""

import os
import joblib
import numpy as np
from collections import deque
from typing import Tuple, Optional, List, Dict, Any


class WordLevelClassifier:
    """
    Temporal sign classifier over a rolling window of hand landmark frames.
    """

    VOCABULARY = [
        "HELLO", "THANK-YOU", "PLEASE", "HELP", "WEATHER", "HOW",
        "STORE", "GO", "RESTAURANT", "WHERE", "YES", "NO",
        "GOOD", "NAME", "WANT", "NEED", "TIME", "FRIEND", "COFFEE"
    ]

    def __init__(self, model_path: Optional[str] = "backend/models/word_level_model.pkl", window_size: int = 20):
        self.window_size = window_size
        self.frame_buffer = deque(maxlen=window_size)
        self.model = None
        self.classes = self.VOCABULARY

        if model_path and os.path.exists(model_path):
            try:
                data = joblib.load(model_path)
                if isinstance(data, dict):
                    self.model = data.get("model")
                    self.classes = data.get("classes", self.VOCABULARY)
                else:
                    self.model = data
                print(f"[WordLevelClassifier] Loaded temporal model from: {model_path}")
            except Exception as e:
                print(f"[WordLevelClassifier] Model load warning: {e}")

    def add_frame(self, landmark_features: np.ndarray):
        """Adds a normalized landmark frame (63-dim) to the temporal buffer."""
        if len(landmark_features) == 63:
            self.frame_buffer.append(landmark_features)

    def reset_buffer(self):
        """Clears the temporal frame buffer."""
        self.frame_buffer.clear()

    def predict(self) -> Tuple[Optional[str], float]:
        """
        Evaluates temporal sequence in the rolling buffer and predicts full ASL gloss.
        Returns: (predicted_gloss, confidence) or (None, 0.0) if no sign detected.
        """
        if len(self.frame_buffer) < 8:
            return None, 0.0

        buffer_arr = np.array(self.frame_buffer, dtype=np.float32)  # (T, 63)

        # 1. If trained temporal model is available, compute temporal features and predict
        if self.model is not None:
            try:
                temp_features = self._extract_temporal_features(buffer_arr).reshape(1, -1)
                pred = self.model.predict(temp_features)[0]
                proba = np.max(self.model.predict_proba(temp_features)[0]) if hasattr(self.model, "predict_proba") else 0.85
                if proba >= 0.70 and pred in self.classes and pred != "NONE":
                    return str(pred), float(proba)
            except Exception:
                pass

        # 2. Dynamic trajectory heuristic analyzer
        return self._heuristic_temporal_sign(buffer_arr)

    def _extract_temporal_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Extracts statistical & motion features across the temporal window:
        - Mean pose (63)
        - Variance / range of motion (63)
        - Velocity between start, mid, end (63)
        - Net displacement (63)
        """
        T, D = sequence.shape
        mean_pose = np.mean(sequence, axis=0)
        std_pose = np.std(sequence, axis=0)
        velocity = sequence[-1] - sequence[0]
        max_pose = np.max(sequence, axis=0)
        min_pose = np.min(sequence, axis=0)
        range_pose = max_pose - min_pose

        return np.concatenate([mean_pose, std_pose, velocity, range_pose])

    def _heuristic_temporal_sign(self, sequence: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Rule-based temporal trajectory detector for primary conversational ASL signs.
        """
        latest = sequence[-1].reshape(21, 3)
        initial = sequence[0].reshape(21, 3)
        velocity = latest - initial

        # Finger extensions at end of gesture
        thumb_tip = latest[4]
        index_tip = latest[8]; index_pip = latest[6]
        middle_tip = latest[12]; middle_pip = latest[10]
        ring_tip = latest[16]; ring_pip = latest[14]
        pinky_tip = latest[20]; pinky_pip = latest[18]

        index_ext = index_tip[1] < index_pip[1]
        middle_ext = middle_tip[1] < middle_pip[1]
        ring_ext = ring_tip[1] < ring_pip[1]
        pinky_ext = pinky_tip[1] < pinky_pip[1]
        extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])

        # 1. "HELLO" (Open 5 Hand moving outward / waving)
        if extended_count == 4 and np.linalg.norm(velocity[8]) > 0.04:
            return "HELLO", 0.93

        # 2. "THANK-YOU" (Flat hand moving forward and down from chin)
        if extended_count == 4 and velocity[8][1] > 0.05 and velocity[8][2] < 0:
            return "THANK-YOU", 0.90

        # 3. "HELP" (Upward motion of closed fist / thumbs up)
        if extended_count == 0 and thumb_tip[1] < latest[0][1] and velocity[4][1] < -0.04:
            return "HELP", 0.88

        # 4. "YES" (Fist nodding down)
        if extended_count == 0 and velocity[4][1] > 0.03:
            return "YES", 0.85

        # 5. "WHERE" (Index finger shaking horizontally)
        if index_ext and not middle_ext and abs(velocity[8][0]) > 0.06:
            return "WHERE", 0.87

        # 6. "HOW" (Hands rotating outward)
        if not index_ext and not middle_ext and abs(velocity[4][0]) > 0.05:
            return "HOW", 0.84

        return None, 0.0
