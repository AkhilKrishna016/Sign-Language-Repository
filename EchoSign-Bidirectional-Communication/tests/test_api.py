"""
Integration Tests for FastAPI Endpoints
"""

import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.main import app


class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_api_understand_endpoint(self):
        payload = {"raw_input": "TOMORROW WEATHER HOW (0.85)"}
        resp = self.client.post("/api/understand", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "reliable")
        self.assertIn("weather", data["reconstructed_english"].lower())

    def test_api_understand_low_confidence(self):
        payload = {"raw_input": "HELP (0.2) (0.2)"}
        resp = self.client.post("/api/understand", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "repeat")

    def test_api_speech_to_sign_endpoint(self):
        payload = {"english_text": "Where is the restaurant tomorrow?"}
        resp = self.client.post("/api/speech-to-sign", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("TOMORROW", data["asl_gloss"])
        self.assertIn("RESTAURANT", data["asl_gloss"])

    def test_api_dataset_samples_endpoint(self):
        resp = self.client.get("/api/dataset-samples")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("samples", data)

    def test_api_feedback_logs_endpoint(self):
        resp = self.client.get("/api/feedback-logs")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("logs", data)


if __name__ == "__main__":
    unittest.main()
