"""
EchoSign — Streamlit Web Application

Character-by-character ASL sentence builder & communication assistant.
Sign each letter, build words and complete sentences, and explicitly send to the AI model.
Run with:
    streamlit run streamlit_app.py
"""

import os
import sys
import time
import json
import base64
import numpy as np
import cv2
import streamlit as st

sys.path.insert(0, os.path.abspath("."))

from backend.language_engine import SignLanguageUnderstandingEngine
from backend.hand_tracker import HandTracker
from backend.classifiers.fingerspelling_classifier import FingerspellingClassifier
from backend.classifiers.word_level_classifier import WordLevelClassifier
from backend.speech_to_sign import SpeechToSignEngine
from backend.tts_engine import TTSEngine
from backend.feedback_logger import FeedbackLogger

st.set_page_config(
    page_title="EchoSign — ASL Sentence Composer",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State & Engines
@st.cache_resource
def load_engines():
    lang_eng = SignLanguageUnderstandingEngine()
    tracker = HandTracker()
    fs_clf = FingerspellingClassifier()
    word_clf = WordLevelClassifier()
    speech_eng = SpeechToSignEngine()
    tts_eng = TTSEngine()
    fb_logger = FeedbackLogger()
    return lang_eng, tracker, fs_clf, word_clf, speech_eng, tts_eng, fb_logger

lang_engine, hand_tracker, fs_classifier, word_classifier, speech_engine, tts_engine, feedback_logger = load_engines()

if "sentence_words" not in st.session_state:
    st.session_state.sentence_words = [[]]  # List of lists of {char, conf}
if "dialogue_history" not in st.session_state:
    st.session_state.dialogue_history = []

# Sidebar
with st.sidebar:
    st.title("🤟 EchoSign AI")
    st.caption("Character-by-Character ASL Sentence Composer")
    st.markdown("---")

    st.subheader("⚙️ Settings")
    auto_tts = st.toggle("Enable Voice TTS Audio", value=True)

    st.markdown("---")
    st.subheader("📊 Feedback Logs")
    logs = feedback_logger.get_recent_logs(limit=10)
    st.write(f"Logged Uncertainty Turns: **{len(logs)}**")
    if st.button("Clear Feedback Log"):
        feedback_logger.clear_logs()
        st.success("Feedback log cleared.")

# Main Title
st.title("EchoSign — ASL Sentence Composer & Communication Assistant")
st.markdown("Sign hand shapes character-by-character to build words and complete sentences, then send to the AI assistant for natural language understanding and spoken replies.")

tabs = st.tabs(["💬 Sentence Composer Studio", "🧪 Prompt Lab", "🔄 Reverse Speech-to-Sign", "📋 Feedback Logs"])

# ----------------- TAB 1: SENTENCE COMPOSER STUDIO -----------------
with tabs[0]:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🤟 Signer Channel: Character Recognition")
        cam_input = st.camera_input("Sign a character")

        detected_char = None
        detected_conf = 0.0

        if cam_input is not None:
            bytes_data = cam_input.getvalue()
            np_arr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            tracking_res = hand_tracker.process_frame(frame, draw_overlay=True)

            if tracking_res.hands_detected > 0:
                hand = tracking_res.hands[0]
                letter, fs_conf = fs_classifier.predict(hand.normalized_features)
                if letter and letter not in {"nothing"}:
                    detected_char = letter
                    detected_conf = fs_conf
                    st.success(f"Recognized Sign: **{letter}** (Confidence: {fs_conf:.2f}) | Hand: {hand.handedness}")
            else:
                st.info("Position hand in frame to recognize character.")

        st.markdown("### ✍️ Composed Sentence Builder")

        # Display current plain sentence
        plain_words = ["".join(item["char"] for item in w) for w in st.session_state.sentence_words]
        current_sentence_str = " ".join(w for w in plain_words if w)

        st.info(f"**Current Sentence:** `{current_sentence_str or '[Empty - Add letters below]'}`")

        # Action Buttons
        btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)
        with btn_c1:
            if st.button("➕ Add Recognized Letter", type="secondary", disabled=(detected_char is None)):
                if detected_char:
                    if detected_char.lower() == "space" or detected_char == " ":
                        st.session_state.sentence_words.append([])
                    elif detected_char.lower() == "del":
                        if st.session_state.sentence_words[-1]:
                            st.session_state.sentence_words[-1].pop()
                    else:
                        st.session_state.sentence_words[-1].append({"char": detected_char, "conf": round(detected_conf, 2)})
                    st.rerun()

        with btn_c2:
            if st.button("␣ Add Space (New Word)"):
                if st.session_state.sentence_words[-1]:
                    st.session_state.sentence_words.append([])
                    st.rerun()

        with btn_c3:
            if st.button("⌫ Backspace"):
                if st.session_state.sentence_words[-1]:
                    st.session_state.sentence_words[-1].pop()
                elif len(st.session_state.sentence_words) > 1:
                    st.session_state.sentence_words.pop()
                st.rerun()

        with btn_c4:
            if st.button("🗑️ Clear Sentence"):
                st.session_state.sentence_words = [[]]
                st.rerun()

        # Prominent Send Button
        st.markdown("---")
        if st.button("🚀 SEND SENTENCE TO MODEL", type="primary", use_container_width=True):
            active_words = [w for w in st.session_state.sentence_words if len(w) > 0]
            if not active_words:
                st.warning("Please add some characters first before sending.")
            else:
                structured_parts = []
                for word in active_words:
                    tokens_str = "-".join(f"{item['char']}({item['conf']})" for item in word)
                    structured_parts.append(f"[FS] {tokens_str}")
                structured_payload = " ".join(structured_parts)

                res = lang_engine.parse_turn(structured_payload)
                feedback_logger.log_turn(res)

                st.session_state.dialogue_history.append({
                    "signer": res.reconstructed_english or current_sentence_str,
                    "reply": res.response_text,
                    "status": res.status,
                    "gloss": res.gloss_reply
                })
                # Reset sentence composer
                st.session_state.sentence_words = [[]]
                st.rerun()

    with col2:
        st.subheader("🗣️ Listener Channel: Captions & Spoken Reply")

        if st.session_state.dialogue_history:
            last_entry = st.session_state.dialogue_history[-1]
            st.info(f"**Live Captions:** \"{last_entry['reply']}\"")
            st.caption(f"ASL Gloss: `{last_entry['gloss']}` | Status: **{last_entry['status'].upper()}**")

            if auto_tts:
                audio_bytes = tts_engine.synthesize_to_bytes(last_entry["reply"])
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
        else:
            st.write("*Waiting for signed sentence from partner...*")

        st.markdown("#### 📜 Dialogue Transcript")
        for item in reversed(st.session_state.dialogue_history):
            st.markdown(f"**Signer:** {item['signer']}")
            st.markdown(f"**Assistant:** {item['reply']}")
            st.markdown("---")

# ----------------- TAB 2: PROMPT LAB -----------------
with tabs[1]:
    st.subheader("🧪 Sign Language Understanding Layer — Prompt Lab")
    preset_options = [
        "TOMORROW WEATHER HOW (0.85)",
        "[FS] R(0.9)-E(0.88)-S(0.55)-T(0.91)-A(0.93)-U(0.87)-R(0.9)-A(0.92)-N(0.6)-T(0.94) NEAR WHERE (0.8)",
        "HELP (0.3) (0.2)",
        "[FS] J(0.88)-O(0.95)-H(0.61)-N(0.93)",
        "NAME YOU WHAT (0.92)",
        "[FS] C-O-F-F-E-E WANT (0.88)"
    ]

    selected = st.selectbox("Presets:", ["Custom..."] + preset_options)
    prompt_in = selected if selected != "Custom..." else "TOMORROW WEATHER HOW (0.85)"
    user_in = st.text_area("Structured Input:", value=prompt_in, height=80)

    if st.button("▶️ Execute Prompt"):
        res = lang_engine.parse_turn(user_in)
        feedback_logger.log_turn(res)
        st.markdown(f"**Reconstructed English:** `{res.reconstructed_english}`")
        st.markdown(f"**Reply:** \"{res.response_text}\"")
        st.markdown(f"**Status:** `{res.status}` (Avg: {res.average_confidence:.2f})")

# ----------------- TAB 3: REVERSE SPEECH-TO-SIGN -----------------
with tabs[2]:
    st.subheader("🔄 Reverse Channel: English ➔ ASL Cards")
    eng_text = st.text_input("Enter message for deaf user:", value="Where is the restaurant?")
    if st.button("Convert to ASL"):
        s_res = speech_engine.english_to_asl(eng_text)
        st.markdown(f"**ASL Gloss:** `{s_res.asl_gloss}`")
        cols = st.columns(min(len(s_res.visual_tokens), 5)) if s_res.visual_tokens else []
        for i, tok in enumerate(s_res.visual_tokens):
            with cols[i % len(cols)]:
                st.button(f"🤟 {tok.text}", help=tok.description, key=f"tok_{i}")
                st.caption(tok.description)

# ----------------- TAB 4: FEEDBACK LOGS -----------------
with tabs[3]:
    st.subheader("📋 Low-Confidence & Uncertainty Feedback Log")
    logs = feedback_logger.get_recent_logs(limit=50)
    if logs:
        st.json(logs)
    else:
        st.info("No logs recorded yet.")
