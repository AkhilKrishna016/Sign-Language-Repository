"""
Training Script: Word-Level Temporal Sign Classifier

Trains a temporal sequence model (MLP / Random Forest over rolling window trajectory features)
on WLASL temporal landmark sequences. Saves model to backend/models/word_level_model.pkl.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_word_level_model(
    npz_path: str = "data/wlasl_sequences.npz",
    output_path: str = "backend/models/word_level_model.pkl"
):
    print("=" * 60)
    print("  Training Word-Level Temporal Sign Classifier (WLASL)")
    print("=" * 60)

    if not os.path.exists(npz_path):
        print(f"[Training] Preprocessed sequence data not found at {npz_path}. Running preprocessor...")
        from scripts.preprocess_wlasl import preprocess_wlasl_dataset
        preprocess_wlasl_dataset(output_npz=npz_path)

    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    classes = data["classes"]

    print(f"[Training] Loaded {len(X)} temporal sequences across {len(classes)} sign classes.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n[Training] Fitting Temporal Trajectory Classifier (MLP 256, 128)...")
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400, random_state=42, early_stopping=True)
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_test)
    val_acc = accuracy_score(y_test, val_preds)
    print(f"  -> Temporal Model Validation Accuracy: {val_acc * 100:.2f}%")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({"model": clf, "classes": list(clf.classes_)}, output_path)
    print(f"  -> Saved Word-Level Model to: {output_path}")

    print("\n" + "=" * 60)
    print("  Word-Level Temporal Training Complete!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    train_word_level_model()
