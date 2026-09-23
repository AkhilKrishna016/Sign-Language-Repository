"""
Turn Segmentation & Utterance Buffering

Detects "hand at rest" pauses to mark when a signed utterance is complete.
Buffers the sequence of recognized word glosses and fingerspelled letters,
and formats them into the exact structured prompt specification:
- Word glosses: WORD(confidence) e.g. TOMORROW WEATHER HOW (0.85)
- Fingerspelled runs: [FS] A(0.9)-B(0.8)...
"""

import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class BufferedToken:
    symbol: str
    confidence: float
    is_fingerspelled: bool
    timestamp: float


@dataclass
class StructuredTurnPayload:
    structured_text: str
    tokens: List[BufferedToken]
    duration_sec: float
    emitted_at: float


class TurnSegmenter:
    """
    Tracks temporal stream of gestures and emits structured turns on pause boundaries.
    """

    def __init__(self, pause_threshold_sec: float = 1.3, hold_frames_threshold: int = 3):
        self.pause_threshold = pause_threshold_sec
        self.hold_frames_threshold = hold_frames_threshold

        self.tokens_buffer: List[BufferedToken] = []
        self.last_active_time = time.time()
        self.turn_start_time = time.time()

        self.current_candidate = None
        self.candidate_type = "none"
        self.candidate_confidence = 0.0
        self.consecutive_frames = 0

    def feed_frame_recognition(
        self,
        symbol: Optional[str],
        symbol_type: str,  # "letter", "gloss", "none"
        confidence: float,
        hand_present: bool
    ) -> Optional[StructuredTurnPayload]:
        """
        Feeds recognition from current frame.
        Returns StructuredTurnPayload if a turn boundary was crossed, otherwise None.
        """
        now = time.time()

        if not hand_present or not symbol or symbol in {"nothing", "none"}:
            # Hand is at rest or not visible
            if self.tokens_buffer and (now - self.last_active_time) > self.pause_threshold:
                return self.emit_turn()
            return None

        # Hand is active: apply temporal debouncing
        if confidence >= 0.45:
            if symbol == self.current_candidate:
                self.consecutive_frames += 1
                self.candidate_confidence = max(self.candidate_confidence, confidence)
            else:
                self.current_candidate = symbol
                self.candidate_type = symbol_type
                self.candidate_confidence = confidence
                self.consecutive_frames = 1

            if self.consecutive_frames == self.hold_frames_threshold:
                # Confirm token into buffer
                if not self.tokens_buffer:
                    self.turn_start_time = now

                is_fs = (symbol_type == "letter")
                
                # Check for space / deletion
                if symbol == "del":
                    if self.tokens_buffer:
                        self.tokens_buffer.pop()
                elif symbol == " ":
                    pass  # Spaces delimit fingerspelled words
                else:
                    self.tokens_buffer.append(BufferedToken(
                        symbol=symbol,
                        confidence=round(self.candidate_confidence, 2),
                        is_fingerspelled=is_fs,
                        timestamp=now
                    ))

                self.last_active_time = now

        if self.tokens_buffer and (now - self.last_active_time) > self.pause_threshold:
            return self.emit_turn()

        return None

    def emit_turn(self) -> Optional[StructuredTurnPayload]:
        """
        Compiles buffered tokens into structured prompt text and resets the buffer.
        """
        if not self.tokens_buffer:
            return None

        parts = []
        fs_buffer = []

        for token in self.tokens_buffer:
            if token.is_fingerspelled:
                fs_buffer.append(f"{token.symbol}({token.confidence})")
            else:
                if fs_buffer:
                    parts.append(f"[FS] {'-'.join(fs_buffer)}")
                    fs_buffer = []
                parts.append(f"{token.symbol} ({token.confidence})")

        if fs_buffer:
            parts.append(f"[FS] {'-'.join(fs_buffer)}")

        structured_text = " ".join(parts)
        now = time.time()
        payload = StructuredTurnPayload(
            structured_text=structured_text,
            tokens=list(self.tokens_buffer),
            duration_sec=round(now - self.turn_start_time, 2),
            emitted_at=now
        )

        self.reset()
        return payload

    def reset(self):
        """Clears active turn state."""
        self.tokens_buffer = []
        self.current_candidate = None
        self.candidate_type = "none"
        self.candidate_confidence = 0.0
        self.consecutive_frames = 0
        self.last_active_time = time.time()
        self.turn_start_time = time.time()
