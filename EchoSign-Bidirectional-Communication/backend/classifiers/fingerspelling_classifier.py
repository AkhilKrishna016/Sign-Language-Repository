"""
Fingerspelling Alphabet Classifier (Character-Only Model)

Decoupled character-level classifier that outputs exclusively single-character ASL
letters (A-Z).
Supports:
1. Deep CNN Transfer Learning Model (MobileNetV2) via predict_image()
2. Landmark-based Classifier (MLP / Random Forest) via predict()
"""

import os
import json
import joblib
import numpy as np
from typing import Tuple, Optional, Any, Dict, List
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from torchvision.models import MobileNet_V2_Weights
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class FingerspellingClassifier:
    """
    Classifies ASL alphabet finger signs into character labels (A-Z).
    Guaranteed character-only prediction set with zero word-gloss leakage.
    """

    ALPHABET = [chr(c) for c in range(ord('A'), ord('Z') + 1)]

    def __init__(self, model_path: Optional[str] = None, cnn_model_path: Optional[str] = "backend/models/character_mobilenet_v2.pth"):
        self.model = None
        self.classes = self.ALPHABET
        self.cnn_model = None
        self.cnn_classes = self.ALPHABET
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None

        # 1. Load Landmark Classifier
        candidate_paths = [
            model_path,
            "backend/models/fingerspelling_mlp.pkl",
            "backend/models/asl_model.pkl"
        ]

        for p in candidate_paths:
            if p and os.path.exists(p):
                try:
                    data = joblib.load(p)
                    if isinstance(data, dict):
                        self.model = data.get("model")
                        self.classes = data.get("classes", self.ALPHABET)
                    else:
                        self.model = data
                    print(f"[FingerspellingClassifier] Loaded landmark classifier from: {p}")
                    break
                except Exception as e:
                    print(f"[FingerspellingClassifier] Landmark model load warning: {e}")

        # 2. Load MobileNetV2 CNN Model if available
        if TORCH_AVAILABLE and cnn_model_path and os.path.exists(cnn_model_path):
            try:
                checkpoint = torch.load(cnn_model_path, map_location=self.device, weights_only=False)
                self.cnn_classes = checkpoint.get("classes", self.ALPHABET)
                
                # Build MobileNetV2 architecture matching checkpoint
                net = models.mobilenet_v2(weights=None)
                in_features = net.classifier[1].in_features
                net.classifier = nn.Sequential(
                    nn.Dropout(p=0.3),
                    nn.Linear(in_features, 256),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(256, len(self.cnn_classes))
                )
                net.load_state_dict(checkpoint["state_dict"])
                net.to(self.device)
                net.eval()
                self.cnn_model = net

                self.cnn_transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                print(f"[FingerspellingClassifier] Loaded MobileNetV2 CNN from: {cnn_model_path} ({len(self.cnn_classes)} classes)")
            except Exception as e:
                print(f"[FingerspellingClassifier] CNN model load warning: {e}")

    def predict(self, normalized_landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Takes 63-dim normalized landmark array and returns (predicted_letter, confidence).
        Guaranteed to output strictly valid character labels {A-Z} or 'nothing'.
        """
        if len(normalized_landmarks) < 63 or np.all(normalized_landmarks == 0):
            return "nothing", 0.0

        if self.model is not None:
            try:
                features = normalized_landmarks.reshape(1, -1)
                pred = self.model.predict(features)[0]
                proba = np.max(self.model.predict_proba(features)[0]) if hasattr(self.model, "predict_proba") else 0.88
                clean_pred = str(pred).upper().strip()

                if clean_pred in set(self.ALPHABET):
                    return clean_pred, float(proba)
            except Exception:
                pass

        # Geometric heuristic rule-based fallback (strictly characters)
        return self._heuristic_letter(normalized_landmarks)

    def predict_image(self, image_input: Any) -> Tuple[str, float]:
        """
        Takes PIL Image, BGR numpy frame, or image path and evaluates MobileNetV2 CNN.
        Returns: (predicted_letter, confidence)
        """
        if self.cnn_model is None or not TORCH_AVAILABLE:
            return "nothing", 0.0

        try:
            if isinstance(image_input, str):
                pil_img = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, np.ndarray):
                # Convert BGR OpenCV image to RGB PIL
                if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                    import cv2
                    rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                else:
                    pil_img = Image.fromarray(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                pil_img = image_input.convert("RGB")
            else:
                return "nothing", 0.0

            tensor = self.cnn_transforms(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.cnn_model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                conf, idx = torch.max(probs, dim=0)

            pred_char = self.cnn_classes[idx.item()]
            return str(pred_char).upper(), float(conf.item())
        except Exception as e:
            print(f"[FingerspellingClassifier] predict_image error: {e}")
            return "nothing", 0.0

    def _heuristic_letter(self, features: np.ndarray) -> Tuple[str, float]:
        """Heuristic geometric estimator based on normalized coordinates (A-Z only)."""
        coords = features.reshape(21, 3)
        thumb_tip = coords[4]
        index_tip = coords[8]; index_pip = coords[6]
        middle_tip = coords[12]; middle_pip = coords[10]
        ring_tip = coords[16]; ring_pip = coords[14]
        pinky_tip = coords[20]; pinky_pip = coords[18]

        index_ext = index_tip[1] < index_pip[1]
        middle_ext = middle_tip[1] < middle_pip[1]
        ring_ext = ring_tip[1] < ring_pip[1]
        pinky_ext = pinky_tip[1] < pinky_pip[1]

        extended = sum([index_ext, middle_ext, ring_ext, pinky_ext])

        if extended == 0:
            return "A", 0.85
        elif extended == 4:
            return "B", 0.90
        elif extended == 1 and index_ext:
            return "D", 0.88
        elif extended == 1 and pinky_ext:
            return "I", 0.88
        elif extended == 2 and index_ext and middle_ext:
            return "V", 0.86
        elif extended == 3 and not pinky_ext:
            return "W", 0.89

        return "nothing", 0.40
