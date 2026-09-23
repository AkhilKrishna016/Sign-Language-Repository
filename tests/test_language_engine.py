"""
Unit Tests for SignLanguageUnderstandingEngine
"""

import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.language_engine import SignLanguageUnderstandingEngine


class TestLanguageEngineRules(unittest.TestCase):
    def setUp(self):
        self.engine = SignLanguageUnderstandingEngine(enable_logging=False)

    def test_worked_example_1_weather(self):
        res = self.engine.parse_turn("TOMORROW WEATHER HOW (0.85)")
        self.assertEqual(res.status, "reliable")
        self.assertGreaterEqual(res.average_confidence, 0.75)
        self.assertIn("weather", res.reconstructed_english.lower())
        self.assertIn("tomorrow", res.reconstructed_english.lower())

    def test_worked_example_2_silent_corrections(self):
        # M confused with N, A confused with S
        raw_text = "[FS] R(0.9)-E(0.88)-A(0.55)-T(0.91)-A(0.93)-U(0.87)-R(0.9)-A(0.92)-M(0.6)-T(0.94) NEAR WHERE (0.8)"
        res = self.engine.parse_turn(raw_text)
        self.assertEqual(res.status, "reliable")
        self.assertIn("restaurant", res.reconstructed_english.lower())
        self.assertGreater(len(res.applied_corrections), 0)

    def test_worked_example_3_low_confidence(self):
        res = self.engine.parse_turn("HELP (0.3) (0.2)")
        self.assertEqual(res.status, "repeat")
        self.assertEqual(res.response_text, "I didn't catch that clearly — could you sign that again?")

    def test_fingerspelled_name(self):
        res = self.engine.parse_turn("[FS] J(0.88)-O(0.95)-H(0.61)-N(0.93)")
        self.assertEqual(res.status, "reliable")
        self.assertIn("John", res.reconstructed_english)

    def test_topic_comment_grammar(self):
        res = self.engine.parse_turn("STORE YOU GO (0.88)")
        self.assertEqual(res.status, "reliable")
        self.assertEqual(res.reconstructed_english, "Are you going to the store?")


if __name__ == "__main__":
    unittest.main()
