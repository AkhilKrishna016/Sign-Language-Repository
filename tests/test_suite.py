"""
Unit Tests for SignLanguageUnderstandingEngine (unittest runner)
"""

import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.language_engine import SignLanguageUnderstandingEngine
from fastapi.testclient import TestClient
from backend.main import app

class TestLanguageEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SignLanguageUnderstandingEngine()
        self.client = TestClient(app)

    def test_worked_example_1_weather(self):
        res = self.engine.parse_turn("TOMORROW WEATHER HOW (0.85)")
        self.assertEqual(res.status, "reliable")
        self.assertGreaterEqual(res.average_confidence, 0.75)
        self.assertIn("weather", res.reconstructed_english.lower())
        self.assertIn("tomorrow", res.reconstructed_english.lower())
        self.assertTrue("cloudy" in res.response_text.lower() or "rain" in res.response_text.lower() or "75" in res.response_text)

    def test_worked_example_2_silent_correction(self):
        res = self.engine.parse_turn("[FS] R(0.9)-E(0.88)-S(0.55)-T(0.91)-A(0.93)-U(0.87)-R(0.9)-A(0.92)-N(0.6)-T(0.94) NEAR WHERE (0.8)")
        self.assertEqual(res.status, "reliable")
        self.assertIn("restaurant", res.reconstructed_english.lower())
        self.assertIn("where", res.reconstructed_english.lower())
        self.assertTrue(any("RESTAURANT" in token.text for token in res.tokens))

    def test_worked_example_3_low_confidence(self):
        res = self.engine.parse_turn("HELP (0.3) (0.2)")
        self.assertEqual(res.status, "repeat")
        self.assertTrue(res.has_low_confidence)
        self.assertEqual(res.response_text, "I didn't catch that clearly — could you sign that again?")

    def test_fingerspelled_name(self):
        res = self.engine.parse_turn("[FS] J(0.88)-O(0.95)-H(0.61)-N(0.93)")
        self.assertEqual(res.status, "reliable")
        self.assertIn("John", res.reconstructed_english)

    def test_topic_fronting(self):
        res = self.engine.parse_turn("STORE YOU GO (0.9)")
        self.assertEqual(res.status, "reliable")
        self.assertEqual(res.reconstructed_english, "Are you going to the store?")

    def test_api_understand(self):
        resp = self.client.post("/api/understand", json={"raw_input": "TOMORROW WEATHER HOW (0.85)"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "reliable")

    def test_api_speech_to_sign(self):
        resp = self.client.post("/api/speech-to-sign", json={"english_text": "Where is the restaurant tomorrow?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("TOMORROW", data["asl_gloss"])
        self.assertIn("RESTAURANT", data["asl_gloss"])

    def test_api_dataset_samples(self):
        resp = self.client.get("/api/dataset-samples")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("samples", data)
        self.assertGreater(len(data["samples"]), 0)


if __name__ == "__main__":
    unittest.main()
