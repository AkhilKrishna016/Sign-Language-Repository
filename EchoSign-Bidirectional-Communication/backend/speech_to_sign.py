"""
Speech to Sign Language Engine (Reverse Communication Channel)

Translates English speech/text into ASL Gloss sequences and visual sign/fingerspelling cards
for the deaf/mute recipient.
"""

import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False


@dataclass
class VisualSignToken:
    text: str
    token_type: str  # "gloss", "fingerspell", "punctuation"
    fingerspell_letters: List[str] = field(default_factory=list)
    description: str = ""
    icon: str = "sign-language"


@dataclass
class EnglishToSignResult:
    original_english: str
    asl_gloss: str
    visual_tokens: List[VisualSignToken]
    estimated_duration_sec: float


class SpeechToSignEngine:
    """
    Translates spoken or typed English to ASL Gloss and visual sign cues.
    """

    # Common Sign Dictionary with visual metadata
    SIGN_DICTIONARY = {
        "HELLO": {"desc": "Open hand near forehead, wave outward", "icon": "hand-wave"},
        "HI": {"desc": "Casual open hand wave", "icon": "hand-wave"},
        "THANK-YOU": {"desc": "Flat fingers touch chin, move forward toward person", "icon": "heart-handshake"},
        "THANKS": {"desc": "Flat fingers touch chin, move forward toward person", "icon": "heart-handshake"},
        "PLEASE": {"desc": "Open flat palm rubs circle on chest", "icon": "sparkles"},
        "HELP": {"desc": "Closed fist (thumbs up) on flat open palm, lift up together", "icon": "hands-helping"},
        "WEATHER": {"desc": "W-hands facing each other, twisting downwards like wind", "icon": "cloud-sun"},
        "TOMORROW": {"desc": "A-hand thumb moves forward along jawline", "icon": "calendar-arrow-right"},
        "TODAY": {"desc": "Y-hands move down twice in front of body", "icon": "calendar-check"},
        "NOW": {"desc": "Y-hands drop down sharply once", "icon": "clock"},
        "STORE": {"desc": "Flat-O hands held near chest, shake forward twice", "icon": "store"},
        "GO": {"desc": "Both index fingers arc forward away from body", "icon": "arrow-right-circle"},
        "COME": {"desc": "Both index fingers arc inward toward body", "icon": "arrow-left-circle"},
        "RESTAURANT": {"desc": "R-hand rubs down both sides of mouth/chin", "icon": "utensils"},
        "HOSPITAL": {"desc": "H-hand draws cross on opposite upper arm", "icon": "hospital"},
        "WHERE": {"desc": "Index finger held up, shake side to side with furrowed brow", "icon": "map-pin-question"},
        "WHAT": {"desc": "Open palms facing up, shake side to side with furrowed brow", "icon": "help-circle"},
        "WHEN": {"desc": "Index finger circles and touches tip of other index finger", "icon": "clock"},
        "WHY": {"desc": "Touch forehead with flat hand, pull into Y-hand", "icon": "help-circle"},
        "HOW": {"desc": "Curved hands back to back, roll outward with raised brows", "icon": "sparkles"},
        "YES": {"desc": "S-fist nods up and down like a head nodding", "icon": "check-circle"},
        "NO": {"desc": "Index and middle fingers snap down onto thumb", "icon": "x-circle"},
        "GOOD": {"desc": "Flat hand touches chin, moves down onto other flat palm", "icon": "thumbs-up"},
        "BAD": {"desc": "Flat hand touches chin, turns downward away", "icon": "thumbs-down"},
        "NAME": {"desc": "H-fingers of both hands tap each other twice in an X shape", "icon": "id-badge"},
        "YOU": {"desc": "Index finger points directly at conversation partner", "icon": "user"},
        "I": {"desc": "Index finger points to own chest", "icon": "user-check"},
        "ME": {"desc": "Index finger points to own chest", "icon": "user-check"},
        "MY": {"desc": "Flat open palm rests on chest", "icon": "user-check"},
        "YOUR": {"desc": "Flat open palm pushed toward conversation partner", "icon": "user"},
        "WANT": {"desc": "Clawed hands facing up, pull inward toward body", "icon": "sparkles"},
        "NEED": {"desc": "X-finger bends downward firmly", "icon": "alert-circle"},
        "COFFEE": {"desc": "Both S-fists mimic turning a manual coffee grinder", "icon": "coffee"},
        "WATER": {"desc": "W-hand index taps lower lip twice", "icon": "droplet"},
        "MEETING": {"desc": "Open 5 hands move together into flat-O shapes touching", "icon": "users"},
        "TIME": {"desc": "Index finger taps wrist where watch is worn", "icon": "clock"},
        "FOOD": {"desc": "Flat-O hand taps lips twice", "icon": "utensils"},
        "EAT": {"desc": "Flat-O hand brings food to mouth", "icon": "utensils"},
        "DRINK": {"desc": "C-hand mimics holding cup and tipping to mouth", "icon": "glass-water"},
        "FRIEND": {"desc": "Interlocking X-index fingers flip back and forth", "icon": "heart"},
    }

    # Stopwords / Grammatical words dropped in ASL Gloss
    ASL_DROPPED_WORDS = {
        "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "being",
        "to", "of", "in", "at", "by", "for", "with", "about", "against", "into",
        "through", "during", "before", "after", "above", "below", "from", "up",
        "down", "in", "out", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "do", "does", "did", "doing", "would",
        "should", "could", "ought", "i'm", "you're", "he's", "she's", "it's", "we're", "they're"
    }

    def __init__(self):
        self.recognizer = sr.Recognizer() if SR_AVAILABLE else None

    def english_to_asl(self, english_text: str) -> EnglishToSignResult:
        """
        Converts natural English text to ASL Gloss structure and visual sign cards.
        """
        cleaned = english_text.strip()
        if not cleaned:
            return EnglishToSignResult(
                original_english="",
                asl_gloss="",
                visual_tokens=[],
                estimated_duration_sec=0.0
            )

        # Normalize words
        raw_words = re.findall(r"[\w']+|[.,!?;]", cleaned)
        time_tokens = []
        topic_tokens = []
        wh_tokens = []
        comment_tokens = []

        for raw_w in raw_words:
            w_lower = raw_w.lower().replace("'", "")
            w_upper = w_lower.upper()

            # Punctuation
            if raw_w in {".", "!", "?", ",", ";"}:
                continue

            # Check time indicators (ASL puts time first)
            if w_upper in {"TODAY", "TOMORROW", "YESTERDAY", "NOW", "SOON", "LATER", "MORNING", "NIGHT"}:
                time_tokens.append(w_upper)
            # Check WH-question words (ASL puts WH-words at the end)
            elif w_upper in {"WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO", "WHICH"}:
                wh_tokens.append(w_upper)
            # Check dropped function words
            elif w_lower in self.ASL_DROPPED_WORDS:
                continue
            else:
                comment_tokens.append(w_upper)

        # Build ASL Gloss Order: [TIME] + [TOPIC/COMMENT] + [WH-WORD]
        gloss_words = time_tokens + comment_tokens + wh_tokens

        # Fallback if all words were filtered
        if not gloss_words:
            gloss_words = [w.upper() for w in raw_words if re.match(r'^\w+$', w)]

        # Generate Visual Sign Tokens
        visual_tokens: List[VisualSignToken] = []
        for word in gloss_words:
            if word in self.SIGN_DICTIONARY:
                meta = self.SIGN_DICTIONARY[word]
                visual_tokens.append(VisualSignToken(
                    text=word,
                    token_type="gloss",
                    description=meta["desc"],
                    icon=meta["icon"]
                ))
            else:
                # Word requires fingerspelling [FS]
                letters = list(word)
                visual_tokens.append(VisualSignToken(
                    text=f"[FS] {word}",
                    token_type="fingerspell",
                    fingerspell_letters=letters,
                    description=f"Fingerspell: {'-'.join(letters)}",
                    icon="keyboard"
                ))

        asl_gloss_str = " ".join(t.text for t in visual_tokens)
        estimated_duration = len(visual_tokens) * 1.2

        return EnglishToSignResult(
            original_english=cleaned,
            asl_gloss=asl_gloss_str,
            visual_tokens=visual_tokens,
            estimated_duration_sec=round(estimated_duration, 1)
        )

    def transcribe_audio_file(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribes an uploaded audio file to English text using SpeechRecognition.
        """
        if not SR_AVAILABLE or not self.recognizer:
            return None

        try:
            with sr.AudioFile(audio_file_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                return text
        except Exception as e:
            print(f"[STT Error] {e}")
            return None
