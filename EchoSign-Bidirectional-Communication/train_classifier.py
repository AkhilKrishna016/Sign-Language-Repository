"""
ASL Alphabet Landmark Classifier Trainer

Extracts 21 3D hand landmarks from sample images in dataset/character dataset
using MediaPipe Tasks HandLandmarker and trains a Random Forest classifier.
"""

import os
import glob
import time
import joblib
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def init_detector(model_path: str = "backend/models/hand_landmarker.task"):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )
    return vision.HandLandmarker.create_from_options(options)

def extract_landmark_features_from_file(img_path: str, detector):
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
        # Scale normalization by wrist-to-middle MCP distance (index 9)
        scale = np.linalg.norm(coords[9])
        if scale > 1e-5:
            coords /= scale
        return coords.flatten()
    except Exception as e:
        return None

def train_asl_model(data_dir: str, output_model_path: str, samples_per_class: int = 35):
    print(f"[Training] Initializing HandLandmarker from backend/models/hand_landmarker.task...")
    detector = init_detector()

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"[Training] Found {len(classes)} classes: {classes}")

    X = []
    y = []

    start_time = time.time()
    for class_name in classes:
        class_folder = os.path.join(data_dir, class_name)
        img_files = glob.glob(os.path.join(class_folder, "*.jpg")) + glob.glob(os.path.join(class_folder, "*.png"))
        
        sampled_files = img_files[:samples_per_class]
        extracted = 0
        for img_path in sampled_files:
            feats = extract_landmark_features_from_file(img_path, detector)
            if feats is not None:
                X.append(feats)
                y.append(class_name)
                extracted += 1
        
        print(f"  - [{class_name:7s}] Extracted {extracted}/{len(sampled_files)} samples")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print(f"\n[Training] Total dataset: {len(X)} landmark samples across {len(set(y))} classes.")
    if len(X) == 0:
        print("[Training Error] No landmarks extracted.")
        return False

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("[Training] Fitting Random Forest Classifier (100 estimators)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=18, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"\n[Training Result] Validation Accuracy: {val_acc * 100:.2f}%\n")
    print(classification_report(y_val, val_preds))

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump({"model": clf, "classes": list(clf.classes_)}, output_model_path)
    print(f"[Training] Model saved successfully to: {output_model_path}")
    print(f"[Training] Completed in {time.time() - start_time:.2f} seconds.")
    return True

if __name__ == "__main__":
    train_data_path = os.path.join("dataset", "character dataset", "asl_alphabet_train", "asl_alphabet_train")
    model_output = os.path.join("backend", "models", "asl_model.pkl")
    
    if os.path.exists(train_data_path):
        train_asl_model(train_data_path, model_output, samples_per_class=35)
    else:
        print(f"Dataset directory not found at: {train_data_path}")
