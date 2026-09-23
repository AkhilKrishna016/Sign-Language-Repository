"""
Training Script: Fingerspelling Alphabet Classifier

Trains a Multi-Layer Perceptron (MLP) and Random Forest on preprocessed landmark coordinates.
Saves model weights to backend/models/fingerspelling_mlp.pkl and backend/models/asl_model.pkl.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_fingerspelling_models(
    npz_path: str = "data/asl_landmarks.npz",
    output_mlp_path: str = "backend/models/fingerspelling_mlp.pkl",
    output_rf_path: str = "backend/models/asl_model.pkl"
):
    print("=" * 60)
    print("  Training Fingerspelling Classifiers (MLP + Random Forest)")
    print("=" * 60)

    if not os.path.exists(npz_path):
        print(f"[Training] Preprocessed data not found at {npz_path}. Running preprocessor first...")
        from scripts.preprocess_landmarks import preprocess_dataset
        preprocess_dataset(output_npz=npz_path, samples_per_class=40)

    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    classes = data["classes"]

    print(f"[Training] Loaded {len(X)} samples across {len(classes)} classes.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 1. Train MLP Classifier
    print("\n[1/2] Training Lightweight MLP Classifier (128, 64)...")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42, early_stopping=True)
    mlp.fit(X_train, y_train)
    mlp_preds = mlp.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_preds)
    print(f"  -> MLP Validation Accuracy: {mlp_acc * 100:.2f}%")

    os.makedirs(os.path.dirname(output_mlp_path), exist_ok=True)
    joblib.dump({"model": mlp, "classes": list(mlp.classes_)}, output_mlp_path)
    print(f"  -> Saved MLP Model to: {output_mlp_path}")

    # 2. Train Random Forest Classifier
    print("\n[2/2] Training Random Forest Classifier (100 trees)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=18, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"  -> Random Forest Validation Accuracy: {rf_acc * 100:.2f}%")

    joblib.dump({"model": rf, "classes": list(rf.classes_)}, output_rf_path)
    print(f"  -> Saved Random Forest Model to: {output_rf_path}")

    print("\n" + "=" * 60)
    print("  Fingerspelling Training Complete!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    train_fingerspelling_models()
