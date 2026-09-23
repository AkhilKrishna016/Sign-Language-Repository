"""
FastAPI Server for Bidirectional Sign Language Communication System

Provides:
- Language Understanding Layer API (/api/understand)
- Speech-to-Sign Reverse Translation API (/api/speech-to-sign)
- Computer Vision Frame Processing & Landmark Overlay (/api/recognize-frame)
- Text-to-Speech Audio Stream (/api/tts)
- Feedback Loop Logs API (/api/feedback-logs)
- Real-time WebSockets (/ws/stream)
- Dataset Explorer APIs & Static Frontend Delivery
"""

import os
import io
import time
import base64
import json
import cv2
import numpy as np
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.language_engine import SignLanguageUnderstandingEngine, TurnParseResult
from backend.hand_tracker import HandTracker
from backend.classifiers.fingerspelling_classifier import FingerspellingClassifier
from backend.classifiers.word_level_classifier import WordLevelClassifier
from backend.turn_segmentation import TurnSegmenter
from backend.speech_to_sign import SpeechToSignEngine
from backend.tts_engine import TTSEngine
from backend.feedback_logger import FeedbackLogger

app = FastAPI(
    title="EchoSign — Sign Language Communication Assistant",
    description="Language-understanding layer and bidirectional communication system between deaf/mute and hearing partners.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core System Engines
language_engine = SignLanguageUnderstandingEngine()
hand_tracker = HandTracker()
fs_classifier = FingerspellingClassifier()
word_classifier = WordLevelClassifier()
turn_segmenter = TurnSegmenter()
speech_sign_engine = SpeechToSignEngine()
tts_engine = TTSEngine()
feedback_logger = FeedbackLogger()

# Pydantic Schemas
class UnderstandRequest(BaseModel):
    raw_input: str
    custom_vocab: Optional[List[str]] = None

class SpeechToSignRequest(BaseModel):
    english_text: str

class FrameRecognitionRequest(BaseModel):
    image_base64: str

class TTSRequest(BaseModel):
    text: str


@app.post("/api/understand")
def api_understand(req: UnderstandRequest):
    """
    Core Language-Understanding Layer:
    Processes upstream structured CV text (Glosses + Fingerspelling sequences with confidences)
    and returns reconstructed English meaning + conversational partner reply.
    """
    if req.custom_vocab:
        for word in req.custom_vocab:
            language_engine.vocab.add(word.upper())

    result = language_engine.parse_turn(req.raw_input)
    
    # Generate TTS audio as base64 if reliable or clarify/repeat
    audio_b64 = None
    if result.response_text:
        audio_bytes = tts_engine.synthesize_to_bytes(result.response_text)
        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

    return {
        "raw_input": result.raw_input,
        "status": result.status,
        "average_confidence": result.average_confidence,
        "min_confidence": result.min_confidence,
        "has_low_confidence": result.has_low_confidence,
        "reconstructed_english": result.reconstructed_english,
        "response_text": result.response_text,
        "gloss_reply": result.gloss_reply,
        "clarification_options": result.clarification_options,
        "applied_corrections": result.applied_corrections,
        "audio_base64": audio_b64,
        "tokens": [
            {
                "text": t.text,
                "confidence": t.confidence,
                "is_fingerspelled": t.is_fingerspelled,
                "original_letters": t.original_letters
            }
            for t in result.tokens
        ]
    }


@app.post("/api/speech-to-sign")
def api_speech_to_sign(req: SpeechToSignRequest):
    """
    Reverse Communication Channel:
    Converts English speech/text into ASL Gloss and visual sign/fingerspelling sequences.
    """
    result = speech_sign_engine.english_to_asl(req.english_text)
    return {
        "original_english": result.original_english,
        "asl_gloss": result.asl_gloss,
        "estimated_duration_sec": result.estimated_duration_sec,
        "visual_tokens": [
            {
                "text": t.text,
                "token_type": t.token_type,
                "fingerspell_letters": t.fingerspell_letters,
                "description": t.description,
                "icon": t.icon
            }
            for t in result.visual_tokens
        ]
    }


@app.post("/api/recognize-frame")
def api_recognize_frame(req: FrameRecognitionRequest):
    """
    Processes a single base64 webcam frame through modular HandTracker + Classifiers + TurnSegmenter.
    """
    try:
        header_data = req.image_base64
        if "," in header_data:
            header_data = header_data.split(",")[1]
        
        img_bytes = base64.b64decode(header_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # 1. Modular Hand Tracking
        tracking_res = hand_tracker.process_frame(frame_bgr, draw_overlay=True)

        detected_symbol = None
        symbol_type = "none"
        confidence = 0.0
        landmarks = []
        handedness = "Right"

        if tracking_res.hands_detected > 0:
            primary_hand = tracking_res.hands[0]
            landmarks = primary_hand.landmarks
            handedness = primary_hand.handedness
            recognition_mode = getattr(req, "mode", "character") or "character"

            # 2. Evaluate Fingerspelling Classifier (Character-Only)
            letter, fs_conf = fs_classifier.predict(primary_hand.normalized_features)

            if recognition_mode == "word":
                word_classifier.add_frame(primary_hand.normalized_features)
                gloss, gloss_conf = word_classifier.predict()
                if gloss and gloss_conf >= 0.82:
                    detected_symbol = gloss
                    symbol_type = "gloss"
                    confidence = round(gloss_conf, 2)
            else:
                # Default character-only mode: guaranteed strictly A-Z letters
                if letter and letter not in {"nothing", "del"}:
                    detected_symbol = letter
                    symbol_type = "letter"
                    confidence = round(fs_conf, 2)
                else:
                    detected_symbol = None
                    symbol_type = "none"
                    confidence = 0.0

        # 3. Turn Segmentation feed
        emitted_turn = turn_segmenter.feed_frame_recognition(
            symbol=detected_symbol,
            symbol_type=symbol_type,
            confidence=confidence,
            hand_present=tracking_res.hands_detected > 0
        )

        # Encode annotated image
        _, buffer = cv2.imencode('.jpg', tracking_res.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        turn_dialogue_result = None
        if emitted_turn and emitted_turn.structured_text:
            turn_dialogue_result = language_engine.parse_turn(emitted_turn.structured_text)

        return {
            "hand_present": tracking_res.hands_detected > 0,
            "hands_detected": tracking_res.hands_detected,
            "detected_symbol": detected_symbol,
            "symbol_type": symbol_type,
            "confidence": confidence,
            "handedness": handedness,
            "landmarks": landmarks,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            "emitted_turn": {
                "structured_text": emitted_turn.structured_text,
                "reconstructed_english": turn_dialogue_result.reconstructed_english if turn_dialogue_result else None,
                "response_text": turn_dialogue_result.response_text if turn_dialogue_result else None,
                "status": turn_dialogue_result.status if turn_dialogue_result else None,
                "gloss_reply": turn_dialogue_result.gloss_reply if turn_dialogue_result else None,
            } if emitted_turn else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts")
def api_tts(req: TTSRequest):
    """Generates audio MP3 for given text."""
    audio_bytes = tts_engine.synthesize_to_bytes(req.text)
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="Failed to synthesize TTS audio")
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.get("/api/feedback-logs")
def api_get_feedback_logs(limit: int = 50):
    """Returns low-confidence feedback turns for dataset augmentation & model retraining."""
    return {"logs": feedback_logger.get_recent_logs(limit=limit)}


@app.post("/api/clear-feedback-logs")
def api_clear_feedback_logs():
    """Clears the feedback log file."""
    feedback_logger.clear_logs()
    return {"status": "ok", "message": "Feedback log cleared"}


@app.get("/api/dataset-samples")
def api_get_dataset_samples():
    """Returns sample test images from local ASL alphabet dataset."""
    test_dir = os.path.join("dataset", "character dataset", "asl_alphabet_test", "asl_alphabet_test")
    if not os.path.exists(test_dir):
        return {"samples": []}

    files = sorted(os.listdir(test_dir))
    samples = []
    for f in files:
        if f.lower().endswith((".jpg", ".png")):
            label = f.replace("_test.jpg", "").replace("_test.png", "")
            samples.append({"filename": f, "label": label})
    return {"samples": samples}


@app.get("/api/dataset-image/{filename}")
def api_get_dataset_image(filename: str):
    """Serves individual test dataset image."""
    test_dir = os.path.join("dataset", "character dataset", "asl_alphabet_test", "asl_alphabet_test")
    file_path = os.path.join(test_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path, media_type="image/jpeg")


# WebSocket Real-Time Channel
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action")

            if action == "process_frame":
                img_data = msg.get("image")
                if "," in img_data:
                    img_data = img_data.split(",")[1]
                img_bytes = base64.b64decode(img_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame_bgr is not None:
                    tracking_res = hand_tracker.process_frame(frame_bgr, draw_overlay=False)
                    detected_sym = None
                    sym_type = "none"
                    conf = 0.0

                    if tracking_res.hands_detected > 0:
                        hand = tracking_res.hands[0]
                        letter, fs_conf = fs_classifier.predict(hand.normalized_features)
                        mode = msg.get("mode", "character")

                        if mode == "word":
                            word_classifier.add_frame(hand.normalized_features)
                            gloss, gloss_conf = word_classifier.predict()
                            if gloss and gloss_conf >= 0.82:
                                detected_sym = gloss
                                sym_type = "gloss"
                                conf = round(gloss_conf, 2)
                        else:
                            if letter and letter not in {"nothing", "del"}:
                                detected_sym = letter
                                sym_type = "letter"
                                conf = round(fs_conf, 2)

                    emitted_turn = turn_segmenter.feed_frame_recognition(
                        symbol=detected_sym,
                        symbol_type=sym_type,
                        confidence=conf,
                        hand_present=tracking_res.hands_detected > 0
                    )

                    reply_payload = {
                        "type": "frame_result",
                        "hand_present": tracking_res.hands_detected > 0,
                        "hands_detected": tracking_res.hands_detected,
                        "detected_symbol": detected_sym,
                        "symbol_type": sym_type,
                        "confidence": conf,
                        "current_buffer": [f"{t.symbol}({t.confidence})" for t in turn_segmenter.tokens_buffer]
                    }

                    if emitted_turn and emitted_turn.structured_text:
                        turn_res = language_engine.parse_turn(emitted_turn.structured_text)
                        audio_b64 = None
                        if turn_res.response_text:
                            ab = tts_engine.synthesize_to_bytes(turn_res.response_text)
                            if ab:
                                audio_b64 = base64.b64encode(ab).decode('utf-8')

                        reply_payload["emitted_turn"] = {
                            "structured_text": emitted_turn.structured_text,
                            "reconstructed_english": turn_res.reconstructed_english,
                            "response_text": turn_res.response_text,
                            "gloss_reply": turn_res.gloss_reply,
                            "status": turn_res.status,
                            "audio_base64": audio_b64,
                            "applied_corrections": turn_res.applied_corrections
                        }

                    await websocket.send_json(reply_payload)

            elif action == "reset_turn":
                turn_segmenter.reset()
                await websocket.send_json({"type": "turn_reset", "status": "ok"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket Error] {e}")


# Serve Frontend Static Assets
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h1>EchoSign Server Running</h1>")
