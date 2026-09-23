"""
Preprocessing Script: Images -> Landmark Coordinates

Converts ASL Alphabet (and/or Sign Language MNIST) image datasets into normalized 3D hand landmarks
using MediaPipe Hands. Saves feature arrays to data/asl_landmarks.npz for classifier training.
"""

import os
import glob
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def init_landmarker(model_path: str = "backend/models/hand_landmarker.task"):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )
    return vision.HandLandmarker.create_from_options(options)

def extract_landmarks(img_path: str, detector):
    try:
        mp_image = mp.Image.create_from_file(img_path)
        res = detector.detect(mp_image)
        if not res.hand_landmarks:
            return None

        lms = res.hand_landmarks[0]
        wrist = lms[0]
        coords = []
        for lm in lms:
            coords.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

        coords = np.array(coords, dtype=np.float32)
        # Scale normalize by middle MCP distance (index 9)
        scale = np.linalg.norm(coords[9])
        if scale > 1e-5:
            coords /= scale

        return coords.flatten()
    except Exception:
        return None

def preprocess_dataset(
    dataset_dir: str = "dataset/character dataset/asl_alphabet_train/asl_alphabet_train",
    output_npz: str = "data/asl_landmarks.npz",
    samples_per_class: int = 50
):
    print("=" * 60)
    print("  EchoSign - ASL Image -> Landmark Preprocessor")
    print("=" * 60)

    if not os.path.exists(dataset_dir):
        print(f"[Error] Dataset directory not found: {dataset_dir}")
        return False

    detector = init_landmarker()
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    print(f"[Preprocessor] Found {len(classes)} classes in {dataset_dir}")

    X_list = []
    y_list = []

    start_time = time.time()
    for class_name in classes:
        class_folder = os.path.join(dataset_dir, class_name)
        img_files = (
            glob.glob(os.path.join(class_folder, "*.jpg")) +
            glob.glob(os.path.join(class_folder, "*.png"))
        )[:samples_per_class]

        extracted = 0
        for img_p in img_files:
            feats = extract_landmarks(img_p, detector)
            if feats is not None:
                X_list.append(feats)
                y_list.append(class_name)
                extracted += 1

        print(f"  - [{class_name:7s}] Extracted {extracted}/{len(img_files)} landmark vectors")

    detector.close()

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)

    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(output_npz, X=X, y=y, classes=np.array(classes))

    print("=" * 60)
    print(f"  Saved {len(X)} landmark vectors to: {output_npz}")
    print(f"  Total time: {time.time() - start_time:.2f}s")
    print("=" * 60)
    return True

if __name__ == "__main__":
    preprocess_dataset()
