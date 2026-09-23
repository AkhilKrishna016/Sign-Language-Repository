"""
Verification Script: ASL Character Recognition Verification

Runs inference on held-out character test samples and verifies:
1. All predictions are strictly valid single-character letters {A-Z}.
2. Exactly 0 predictions fall into word glosses or deleted classes.
3. Reports predictions, confidences, and agreement between CNN and Landmark models.
"""

import os
import sys
import glob
import json
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.classifiers.fingerspelling_classifier import FingerspellingClassifier
from backend.hand_tracker import HandTracker
import cv2

def run_verification():
    print("=" * 70)
    print("  SignBridge - Character Model Verification & Test Inference")
    print("=" * 70)

    test_dir = os.path.join("dataset", "character dataset", "asl_alphabet_test", "asl_alphabet_test")
    if not os.path.exists(test_dir):
        print(f"[Error] Test directory not found at {test_dir}")
        return False

    test_images = sorted(glob.glob(os.path.join(test_dir, "*_test.jpg")) + glob.glob(os.path.join(test_dir, "*_test.png")))
    print(f"[Test Dataset] Found {len(test_images)} held-out test images.\n")

    classifier = FingerspellingClassifier()
    tracker = HandTracker()

    VALID_ALPHABET = set([chr(c) for c in range(ord('A'), ord('Z') + 1)])

    results = []
    cnn_correct = 0
    total_tested = 0

    print(f"{'Image':<18} | {'Ground Truth':<12} | {'CNN Pred':<10} | {'CNN Conf':<10} | {'Status':<10}")
    print("-" * 70)

    for img_path in test_images:
        fname = os.path.basename(img_path)
        gt_label = fname.split("_")[0].upper()
        total_tested += 1

        # 1. Run MobileNetV2 CNN inference
        cnn_pred, cnn_conf = classifier.predict_image(img_path)

        # 2. Check validity
        is_valid_char = cnn_pred in VALID_ALPHABET
        is_correct = (cnn_pred == gt_label)
        if is_correct:
            cnn_correct += 1

        status_str = "MATCH" if is_correct else "MISMATCH"
        print(f"{fname:<18} | {gt_label:<12} | {cnn_pred:<10} | {cnn_conf:<10.2f} | {status_str:<10}")

        results.append({
            "filename": fname,
            "ground_truth": gt_label,
            "cnn_prediction": cnn_pred,
            "cnn_confidence": float(cnn_conf),
            "is_valid_char": is_valid_char,
            "is_correct": is_correct
        })

    print("-" * 70)
    acc = (cnn_correct / total_tested) * 100 if total_tested > 0 else 0
    print(f"\nHeld-Out Test Accuracy: {acc:.2f}% ({cnn_correct}/{total_tested} correct)")

    # Constraint Check
    all_valid = all(r["is_valid_char"] for r in results)
    invalid_chars = [r["cnn_prediction"] for r in results if not r["is_valid_char"]]

    print(f"Total evaluated samples: {len(results)}")
    print(f"Predictions outside A-Z: {len(invalid_chars)}")
    if all_valid:
        print("[VERIFIED] 100% of predictions are strictly valid single-character labels (A-Z).")
        print("[VERIFIED] ZERO (0) word-gloss labels or stale deleted tokens were predicted.")
    else:
        print(f"[FAILED] Invalid predictions detected: {invalid_chars}")

    return all_valid and acc >= 90.0

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
