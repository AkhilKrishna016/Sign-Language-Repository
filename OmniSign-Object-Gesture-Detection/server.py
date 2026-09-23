"""
Sign & Custom Object Detection Server Backend
Hybrid Neural Inference: SSD MobileNet + MediaPipe Hand Tracking + TFLite Gesture Classifier
"""

import os
import io
import time
import base64
import glob
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Paths
DETECTOR_MODEL_PATH = os.path.join(BASE_DIR, "OOP cp", "CustomObjectDetectionLiveFeedJava-main", "app", "src", "main", "assets", "model2.tflite")
if not os.path.exists(DETECTOR_MODEL_PATH):
    DETECTOR_MODEL_PATH = os.path.join(BASE_DIR, "OOP cp", "model.tflite")

CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, "OOP cp", "converted_tflite (4)", "model.tflite")

# Label Maps
RAW_DETECTOR_LABELS = ["A", "B", "C", "D", "i like you", "Thank You"]
ACTIVE_DETECTOR_LABELS = ["i like you", "Thank You"]
CLASSIFIER_LABELS = ["LikeYou", "ThankYou", "Yes"]
EXCLUDED_CLASSES = {"A", "B", "C", "D"}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35
)

# Initialize FastAPI
app = FastAPI(title="Sign & Gesture Detection Server", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite Interpreters
print(f"[Server] Loading Object Detector from {DETECTOR_MODEL_PATH}...")
try:
    detector_interp = Interpreter(model_path=DETECTOR_MODEL_PATH)
    detector_interp.allocate_tensors()
    det_in_details = detector_interp.get_input_details()
    det_out_details = detector_interp.get_output_details()
    print("[Server] Detector initialized successfully!")
except Exception as e:
    print(f"[Server] Error loading detector: {e}")
    detector_interp = None

print(f"[Server] Loading Classifier from {CLASSIFIER_MODEL_PATH}...")
try:
    classifier_interp = Interpreter(model_path=CLASSIFIER_MODEL_PATH)
    classifier_interp.allocate_tensors()
    clf_in_details = classifier_interp.get_input_details()
    clf_out_details = classifier_interp.get_output_details()
    print("[Server] Classifier initialized successfully!")
except Exception as e:
    print(f"[Server] Error loading classifier: {e}")
    classifier_interp = None


def classify_crop(crop_rgb: np.ndarray, hand_landmarks=None) -> Optional[Dict[str, Any]]:
    if classifier_interp is None or crop_rgb.size == 0:
        return None
    try:
        resized = cv2.resize(crop_rgb, (224, 224)).astype(np.float32) / 255.0
        inp = np.expand_dims(resized, axis=0)
        classifier_interp.set_tensor(clf_in_details[0]['index'], inp)
        classifier_interp.invoke()
        logits = classifier_interp.get_tensor(clf_out_details[0]['index'])[0].copy()
        
        # Softmax probabilities
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        
        # Geometric Hand Pose Booster (MediaPipe 21 3D Landmarks)
        if hand_landmarks and len(hand_landmarks) >= 21:
            wrist = hand_landmarks[0]
            def dist(p1, p2):
                x1 = getattr(p1, 'x', p1.get('x', 0) if isinstance(p1, dict) else 0)
                y1 = getattr(p1, 'y', p1.get('y', 0) if isinstance(p1, dict) else 0)
                x2 = getattr(p2, 'x', p2.get('x', 0) if isinstance(p2, dict) else 0)
                y2 = getattr(p2, 'y', p2.get('y', 0) if isinstance(p2, dict) else 0)
                return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            index_ext = dist(hand_landmarks[8], wrist) > dist(hand_landmarks[6], wrist) * 1.05
            middle_ext = dist(hand_landmarks[12], wrist) > dist(hand_landmarks[10], wrist) * 1.05
            ring_ext = dist(hand_landmarks[16], wrist) > dist(hand_landmarks[14], wrist) * 1.05
            pinky_ext = dist(hand_landmarks[20], wrist) > dist(hand_landmarks[18], wrist) * 1.05
            thumb_ext = dist(hand_landmarks[4], wrist) > dist(hand_landmarks[2], wrist) * 1.05

            num_ext = sum([index_ext, middle_ext, ring_ext, pinky_ext])

            # Class 0: 'LikeYou' -> Index + Pinky extended, Middle + Ring curled
            if index_ext and pinky_ext and not middle_ext and not ring_ext:
                probs[0] += 0.55
            # Class 1: 'ThankYou' -> Flat open hand, 3 or 4 fingers extended straight
            elif (num_ext >= 3 and index_ext and middle_ext) or (index_ext and middle_ext and ring_ext and pinky_ext):
                probs[1] += 0.55
            # Class 2: 'Yes' -> Closed fist, 0 or 1 finger extended (excluding ILY horns)
            elif num_ext <= 1 and not (index_ext and pinky_ext):
                probs[2] += 0.55

            probs = probs / np.sum(probs)

        top_idx = int(np.argmax(probs))
        return {
            "top_label": CLASSIFIER_LABELS[top_idx],
            "top_confidence": round(float(probs[top_idx]) * 100, 1),
            "all_scores": {CLASSIFIER_LABELS[i]: round(float(probs[i]) * 100, 1) for i in range(len(CLASSIFIER_LABELS))}
        }
    except Exception as e:
        print(f"[classify_crop error] {e}")
        return None


def run_object_detection(img_bgr: np.ndarray, score_threshold: float = 0.30) -> Dict[str, Any]:
    t0 = time.perf_counter()
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    detections = []
    
    # 1. Run SSD MobileNet Object Detector (Filtered strictly to gesture classes: i like you, Thank You)
    if detector_interp is not None:
        try:
            resized = cv2.resize(rgb, (320, 320))
            inp_data = np.expand_dims(resized, axis=0).astype(np.uint8)
            detector_interp.set_tensor(det_in_details[0]['index'], inp_data)
            detector_interp.invoke()

            scores = detector_interp.get_tensor(det_out_details[0]['index'])[0]
            boxes = detector_interp.get_tensor(det_out_details[1]['index'])[0]
            count = int(detector_interp.get_tensor(det_out_details[2]['index'])[0])
            classes = detector_interp.get_tensor(det_out_details[3]['index'])[0]

            for i in range(min(count, len(scores))):
                score = float(scores[i])
                if score >= score_threshold:
                    cls_idx = int(classes[i])
                    raw_label = RAW_DETECTOR_LABELS[cls_idx] if 0 <= cls_idx < len(RAW_DETECTOR_LABELS) else f"Class {cls_idx}"
                    
                    # Exclude A, B, C, D entirely
                    if raw_label in EXCLUDED_CLASSES:
                        continue
                    
                    box = boxes[i]
                    ymin, xmin, ymax, xmax = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    
                    px_ymin = int(max(0, ymin * h))
                    px_xmin = int(max(0, xmin * w))
                    px_ymax = int(min(h, ymax * h))
                    px_xmax = int(min(w, xmax * w))

                    detections.append({
                        "label": raw_label,
                        "confidence": round(score * 100, 1),
                        "score": round(score, 4),
                        "source": "ssd_model",
                        "box": {
                            "ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax,
                            "px_ymin": px_ymin, "px_xmin": px_xmin, "px_ymax": px_ymax, "px_xmax": px_xmax
                        }
                    })
        except Exception as e:
            print(f"[Detector error] {e}")

    # 2. Run MediaPipe Hand Tracking & Gesture Classifier
    hand_landmarks_list = []
    clf_res = None
    try:
        mp_res = hands_detector.process(rgb)
        if mp_res.multi_hand_landmarks:
            for hand_landmarks in mp_res.multi_hand_landmarks:
                lms = [{"x": round(lm.x, 3), "y": round(lm.y, 3)} for lm in hand_landmarks.landmark]
                hand_landmarks_list.append(lms)
                
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                
                # Expand box slightly for full hand boundary
                pad_x = (max(xs) - min(xs)) * 0.15
                pad_y = (max(ys) - min(ys)) * 0.15
                xmin = max(0.0, min(xs) - pad_x)
                xmax = min(1.0, max(xs) + pad_x)
                ymin = max(0.0, min(ys) - pad_y)
                ymax = min(1.0, max(ys) + pad_y)
                
                px_ymin = int(max(0, ymin * h))
                px_xmin = int(max(0, xmin * w))
                px_ymax = int(min(h, ymax * h))
                px_xmax = int(min(w, xmax * w))
                
                crop = rgb[px_ymin:px_ymax, px_xmin:px_xmax]
                crop_clf = classify_crop(crop, hand_landmarks.landmark)
                if crop_clf:
                    clf_res = crop_clf
                    
                    label_display = crop_clf["top_label"]
                    if label_display == "LikeYou":
                        label_display = "I Love You"
                    elif label_display == "ThankYou":
                        label_display = "Thank You"
                        
                    detections.append({
                        "label": label_display,
                        "confidence": crop_clf["top_confidence"],
                        "score": round(crop_clf["top_confidence"] / 100.0, 4),
                        "source": "hand_tracker",
                        "box": {
                            "ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax,
                            "px_ymin": px_ymin, "px_xmin": px_xmin, "px_ymax": px_ymax, "px_xmax": px_xmax
                        }
                    })
    except Exception as e:
        print(f"[Hand tracker error] {e}")

    # Full frame classification fallback if no crop classification
    if clf_res is None and classifier_interp is not None:
        first_lms = mp_res.multi_hand_landmarks[0].landmark if (mp_res and mp_res.multi_hand_landmarks) else None
        clf_res = classify_crop(rgb, first_lms)

    # Sort detections by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    inference_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "detections": detections,
        "count": len(detections),
        "landmarks": hand_landmarks_list,
        "inference_ms": inference_ms,
        "classification": clf_res,
        "image_size": {"width": w, "height": h}
    }


@app.get("/api/info")
async def get_model_info():
    return {
        "detector": {
            "loaded": detector_interp is not None,
            "model_path": DETECTOR_MODEL_PATH,
            "labels": ACTIVE_DETECTOR_LABELS,
            "input_size": 320
        },
        "classifier": {
            "loaded": classifier_interp is not None,
            "model_path": CLASSIFIER_MODEL_PATH,
            "labels": CLASSIFIER_LABELS,
            "input_size": 224
        },
        "hand_tracker": {
            "loaded": True,
            "engine": "MediaPipe Solutions Hands"
        }
    }


from fastapi import Request

@app.post("/api/detect")
async def detect_image(request: Request):
    try:
        img_bgr = None
        thresh = 0.30
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            data = await request.json()
            b64_str = data.get("image_base64", "")
            thresh = float(data.get("threshold", 0.30))
            if b64_str:
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                img_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            form = await request.form()
            file = form.get("file")
            b64_str = form.get("image_base64")
            thresh = float(form.get("threshold", 0.30))

            if file and hasattr(file, "read"):
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif b64_str:
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                img_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image data provided"})

        result = run_object_detection(img_bgr, score_threshold=thresh)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/samples")
async def list_sample_images(category: Optional[str] = None):
    # Prioritize user's new custom dataset photos in oops_cp
    custom_patterns = [
        os.path.join(BASE_DIR, "OOP cp", "oops_cp", "i love you", "*.jpg"),
        os.path.join(BASE_DIR, "OOP cp", "oops_cp", "thank you", "*.jpg"),
        os.path.join(BASE_DIR, "OOP cp", "oops_cp", "yes", "*.jpg"),
        os.path.join(BASE_DIR, "OOP cp", "oops_cp", "*", "*.jpg"),
    ]
    archive_patterns = [
        os.path.join(BASE_DIR, "OOP cp", "compressed_b[1]", "*.jpg"),
    ]
    
    custom_files = []
    for pat in custom_patterns:
        for fpath in glob.glob(pat):
            if fpath not in custom_files:
                custom_files.append(fpath)
                
    # Sort custom files by modification time descending (newest photos first)
    custom_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    archive_files = []
    for pat in archive_patterns:
        archive_files.extend(glob.glob(pat))

    all_files = custom_files + archive_files

    category_counts = {}
    samples = []
    
    for idx, fpath in enumerate(all_files):
        rel_path = os.path.relpath(fpath, BASE_DIR).replace("\\", "/")
        fname = os.path.basename(fpath)
        folder = os.path.basename(os.path.dirname(fpath))
        
        category_counts[folder] = category_counts.get(folder, 0) + 1

        if category and category.lower() != "all" and folder.lower() != category.lower():
            continue

        samples.append({
            "id": idx,
            "name": fname,
            "category": folder,
            "is_custom": "oops_cp" in rel_path,
            "modified_time": os.path.getmtime(fpath),
            "url": f"/api/sample_image?path={rel_path}"
        })

    return {
        "samples": samples[:100],
        "total_found": len(all_files),
        "category_counts": category_counts,
        "categories": list(category_counts.keys())
    }


@app.post("/api/upload_sample")
async def upload_dataset_sample(file: UploadFile = File(...), category: str = Form("thank you")):
    try:
        clean_cat = category.strip().lower()
        allowed_cats = {
            "i love you": os.path.join(BASE_DIR, "OOP cp", "oops_cp", "i love you"),
            "thank you": os.path.join(BASE_DIR, "OOP cp", "oops_cp", "thank you"),
            "yes": os.path.join(BASE_DIR, "OOP cp", "oops_cp", "yes"),
        }
        
        target_dir = allowed_cats.get(clean_cat, os.path.join(BASE_DIR, "OOP cp", "oops_cp", clean_cat))
        os.makedirs(target_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_fname = f"WIN_{timestamp}_custom.jpg"
        out_fpath = os.path.join(target_dir, out_fname)
        
        content = await file.read()
        with open(out_fpath, "wb") as f:
            f.write(content)
            
        rel_path = os.path.relpath(out_fpath, BASE_DIR).replace("\\", "/")
        return {
            "success": True,
            "filename": out_fname,
            "category": clean_cat,
            "url": f"/api/sample_image?path={rel_path}"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/sample_image")
async def get_sample_image(path: str):
    abs_path = os.path.normpath(os.path.join(BASE_DIR, path))
    if not abs_path.startswith(BASE_DIR) or not os.path.exists(abs_path):
        return JSONResponse(status_code=404, content={"error": "Sample file not found"})
    return FileResponse(abs_path)



@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            frame_data = data.get("frame")
            threshold = float(data.get("threshold", 0.30))

            if frame_data:
                if "," in frame_data:
                    frame_data = frame_data.split(",")[1]
                img_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img_bgr is not None:
                    res = run_object_detection(img_bgr, score_threshold=threshold)
                    await websocket.send_json(res)
                else:
                    await websocket.send_json({"error": "Failed to decode frame", "detections": []})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Disconnected / error: {e}")


# Serve web frontend
WEB_DIR = os.path.join(BASE_DIR, "web")
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/")
async def root():
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>OmniSign Server is Running!</h1>")
