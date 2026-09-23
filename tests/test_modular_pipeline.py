"""
Unit Tests for Modular Pipeline Components:
- HandTracker
- FingerspellingClassifier
- WordLevelClassifier
- TurnSegmenter
- FeedbackLogger
- SignLanguageUnderstandingEngine
"""

import os
import sys
import unittest
import numpy as np
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.hand_tracker import HandTracker
from backend.classifiers.fingerspelling_classifier import FingerspellingClassifier
from backend.classifiers.word_level_classifier import WordLevelClassifier
from backend.turn_segmentation import TurnSegmenter
from backend.feedback_logger import FeedbackLogger
from backend.language_engine import SignLanguageUnderstandingEngine


class TestModularPipeline(unittest.TestCase):
    def setUp(self):
        self.tracker = HandTracker()
        self.fs_clf = FingerspellingClassifier()
        self.word_clf = WordLevelClassifier()
        self.segmenter = TurnSegmenter(pause_threshold_sec=0.1)
        self.logger = FeedbackLogger(log_path="logs/test_feedback.jsonl")
        self.engine = SignLanguageUnderstandingEngine(enable_logging=False)

    def tearDown(self):
        if os.path.exists("logs/test_feedback.jsonl"):
            os.remove("logs/test_feedback.jsonl")

    def test_fingerspelling_prediction_shape(self):
        dummy_landmarks = np.random.normal(0, 0.2, size=(63,)).astype(np.float32)
        letter, conf = self.fs_clf.predict(dummy_landmarks)
        self.assertIsInstance(letter, str)
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(conf, 0.0)

    def test_word_level_rolling_window(self):
        self.word_clf.reset_buffer()
        for _ in range(15):
            dummy_lms = np.random.normal(0, 0.2, size=(63,)).astype(np.float32)
            self.word_clf.add_frame(dummy_lms)
        gloss, conf = self.word_clf.predict()
        self.assertTrue(gloss is None or isinstance(gloss, str))

    def test_turn_segmentation_emit(self):
        self.segmenter.feed_frame_recognition("A", "letter", 0.90, True)
        self.segmenter.feed_frame_recognition("A", "letter", 0.90, True)
        self.segmenter.feed_frame_recognition("A", "letter", 0.90, True)

        self.segmenter.feed_frame_recognition("B", "letter", 0.85, True)
        self.segmenter.feed_frame_recognition("B", "letter", 0.85, True)
        self.segmenter.feed_frame_recognition("B", "letter", 0.85, True)

        payload = self.segmenter.emit_turn()
        self.assertIsNotNone(payload)
        self.assertIn("[FS]", payload.structured_text)
        self.assertIn("A", payload.structured_text)
        self.assertIn("B", payload.structured_text)

    def test_feedback_logger_writing(self):
        low_conf_turn = self.engine.parse_turn("HELP (0.2) (0.2)")
        self.logger.log_turn(low_conf_turn)

        records = self.logger.get_recent_logs(limit=10)
        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "repeat")
        self.assertIn("Low confidence", records[0]["reason"])

    def test_silent_confusion_corrections(self):
        # M confused with N, A confused with S
        raw_text = "[FS] R(0.9)-E(0.88)-A(0.55)-T(0.91)-A(0.93)-U(0.87)-R(0.9)-A(0.92)-M(0.6)-T(0.94) NEAR WHERE (0.8)"
        res = self.engine.parse_turn(raw_text)
        self.assertEqual(res.status, "reliable")
        self.assertIn("restaurant", res.reconstructed_english.lower())
        self.assertGreater(len(res.applied_corrections), 0)


if __name__ == "__main__":
    unittest.main()
