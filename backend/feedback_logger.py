"""
Feedback Loop Logger for Sign Language Assistant

Logs low-confidence turns (<0.40), clarifications (0.40-0.75), and repeated turns
to a structured JSON Lines file for future model retraining and fine-tuning.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
FEEDBACK_LOG_PATH = os.path.join(LOG_DIR, "low_confidence_feedback.jsonl")


@dataclass
class FeedbackTurnRecord:
    timestamp: float
    iso_time: str
    raw_input: str
    status: str  # "repeat", "clarify", "reliable"
    average_confidence: float
    min_confidence: float
    reconstructed_english: str
    response_text: str
    reason: str
    tokens: List[Dict[str, Any]]
    applied_corrections: List[str]


class FeedbackLogger:
    def __init__(self, log_path: str = FEEDBACK_LOG_PATH):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_turn(self, turn_result: Any, reason: Optional[str] = None):
        """
        Logs turn if confidence is low, status is repeat or clarify, or if explicitly requested.
        """
        is_low_conf = getattr(turn_result, "has_low_confidence", False) or getattr(turn_result, "min_confidence", 1.0) < 0.75
        status = getattr(turn_result, "status", "reliable")

        if not is_low_conf and status == "reliable" and not reason:
            return  # Skip logging standard high-confidence turns

        if not reason:
            if status == "repeat":
                reason = "Low confidence / repeat requested"
            elif status == "clarify":
                reason = "Medium confidence / ambiguous interpretation"
            else:
                reason = "Sub-optimal confidence"

        tokens_data = []
        for t in getattr(turn_result, "tokens", []):
            if hasattr(t, "text"):
                tokens_data.append({
                    "text": t.text,
                    "confidence": t.confidence,
                    "is_fingerspelled": t.is_fingerspelled
                })
            elif isinstance(t, dict):
                tokens_data.append(t)

        record = FeedbackTurnRecord(
            timestamp=time.time(),
            iso_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            raw_input=getattr(turn_result, "raw_input", ""),
            status=status,
            average_confidence=getattr(turn_result, "average_confidence", 0.0),
            min_confidence=getattr(turn_result, "min_confidence", 0.0),
            reconstructed_english=getattr(turn_result, "reconstructed_english", ""),
            response_text=getattr(turn_result, "response_text", ""),
            reason=reason,
            tokens=tokens_data,
            applied_corrections=getattr(turn_result, "applied_corrections", [])
        )

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        except Exception as e:
            print(f"[FeedbackLogger Error] {e}")

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Reads recent logged feedback records.
        """
        if not os.path.exists(self.log_path):
            return []

        records = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records[-limit:][::-1]  # Return newest first
        except Exception as e:
            print(f"[FeedbackLogger Read Error] {e}")
            return []

    def clear_logs(self):
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
