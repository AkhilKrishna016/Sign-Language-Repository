"""
Text-to-Speech Engine

Generates spoken audio from text replies using gTTS / pyttsx3.
"""

import io
import os
import tempfile
from typing import Optional

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


class TTSEngine:
    def __init__(self, lang: str = "en"):
        self.lang = lang

    def synthesize_to_bytes(self, text: str) -> Optional[bytes]:
        """
        Converts text to MP3 audio bytes.
        """
        if not text.strip():
            return None

        # 1. Try gTTS first (higher natural voice quality)
        if GTTS_AVAILABLE:
            try:
                tts = gTTS(text=text, lang=self.lang, slow=False)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                return fp.read()
            except Exception as e:
                print(f"[TTS] gTTS warning: {e}, falling back to pyttsx3")

        # 2. Try pyttsx3 (offline local synthesis)
        if PYTTSX3_AVAILABLE:
            try:
                engine = pyttsx3.init()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                engine.save_to_file(text, tmp_path)
                engine.runAndWait()
                with open(tmp_path, "rb") as f:
                    audio_bytes = f.read()
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return audio_bytes
            except Exception as e:
                print(f"[TTS] pyttsx3 warning: {e}")

        return None
