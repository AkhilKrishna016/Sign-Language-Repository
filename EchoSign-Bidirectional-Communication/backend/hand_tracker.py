"""
Modular Hand Tracking Pipeline

Extracts 21 3D landmarks (x, y, z) per detected hand (up to 2 hands) using MediaPipe Hands.
Provides normalized coordinate arrays as shared input for downstream classifiers.
"""

import os
import cv2
import numpy as np
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
class SingleHandLandmarks:
    handedness: str  # "Left" or "Right"
    landmarks: List[Dict[str, float]]  # 21 points with x, y, z, px_x, px_y
    raw_coords: np.ndarray             # (21, 3) float32
    normalized_features: np.ndarray    # (63,) float32 scale & wrist-invariant


@dataclass
class HandTrackingResult:
    hands_detected: int
    hands: List[SingleHandLandmarks]
    annotated_frame: np.ndarray


class HandTracker:
    """
    Modular Hand Tracking Engine using MediaPipe Tasks HandLandmarker.
    Extracts landmark coordinates without raw pixel classification.
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

    def __init__(self, task_model_path: str = "backend/models/hand_landmarker.task", max_hands: int = 2):
        self.detector = None
        self.max_hands = max_hands
        self.task_model_path = task_model_path

        if MP_AVAILABLE and os.path.exists(task_model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=task_model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=self.max_hands,
                    min_hand_detection_confidence=0.4,
                    min_hand_presence_confidence=0.4,
                    min_tracking_confidence=0.4
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
            except Exception as e:
                print(f"[HandTracker] Init warning: {e}")

    def process_frame(self, frame_bgr: np.ndarray, draw_overlay: bool = True) -> HandTrackingResult:
        """
        Extracts 21 3D landmarks for each detected hand in frame.
        """
        annotated = frame_bgr.copy()
        if not MP_AVAILABLE or self.detector is None:
            return HandTrackingResult(hands_detected=0, hands=[], annotated_frame=annotated)

        h, w, _ = frame_bgr.shape
        try:
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)

            if not detection_result.hand_landmarks:
                return HandTrackingResult(hands_detected=0, hands=[], annotated_frame=annotated)

            hands_list: List[SingleHandLandmarks] = []

            for idx, hand_lms in enumerate(detection_result.hand_landmarks):
                handedness = "Right"
                if detection_result.handedness and idx < len(detection_result.handedness):
                    handedness = detection_result.handedness[idx][0].category_name

                lm_dicts = []
                raw_coords = []
                for lm in hand_lms:
                    lm_dicts.append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "px_x": lm.x * w,
                        "px_y": lm.y * h
                    })
                    raw_coords.append([lm.x, lm.y, lm.z])

                raw_arr = np.array(raw_coords, dtype=np.float32)

                # Normalize relative to wrist and scale by middle MCP distance
                wrist = raw_arr[self.WRIST]
                norm_coords = raw_arr - wrist
                scale = np.linalg.norm(norm_coords[self.MIDDLE_MCP])
                if scale > 1e-5:
                    norm_coords /= scale
                norm_flat = norm_coords.flatten()

                hands_list.append(SingleHandLandmarks(
                    handedness=handedness,
                    landmarks=lm_dicts,
                    raw_coords=raw_arr,
                    normalized_features=norm_flat
                ))

                # Draw skeleton if requested
                if draw_overlay:
                    color = (0, 230, 255) if handedness == "Right" else (255, 180, 0)
                    for p1, p2 in self.CONNECTIONS:
                        pt1 = (int(lm_dicts[p1]["px_x"]), int(lm_dicts[p1]["px_y"]))
                        pt2 = (int(lm_dicts[p2]["px_x"]), int(lm_dicts[p2]["px_y"]))
                        cv2.line(annotated, pt1, pt2, color, 2, cv2.LINE_AA)

                    for lm in lm_dicts:
                        center = (int(lm["px_x"]), int(lm["px_y"]))
                        cv2.circle(annotated, center, 4, (0, 120, 255), -1, cv2.LINE_AA)
                        cv2.circle(annotated, center, 6, (255, 255, 255), 1, cv2.LINE_AA)

            return HandTrackingResult(
                hands_detected=len(hands_list),
                hands=hands_list,
                annotated_frame=annotated
            )
        except Exception as e:
            return HandTrackingResult(hands_detected=0, hands=[], annotated_frame=annotated)
