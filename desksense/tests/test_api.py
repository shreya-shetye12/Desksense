import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from desksense.api import app


class FakePredictor:
    def __init__(self, *args):
        self.classes, self.device, self.last_path = ["a", "b"], "cpu", None

    def sliding(self, path, *args):
        self.last_path = Path(path)
        if self.last_path.read_bytes() == b"invalid":
            raise ValueError("Invalid video")
        return {"predictions": [{"label": "a"}], "classes": self.classes}


class APITests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"DESKSENSE_RUN": "fake"})
        self.predictor = patch("desksense.api.Predictor", FakePredictor)
        self.env.start()
        self.predictor.start()
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.predictor.stop()
        self.env.stop()

    def test_health(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ready")

    def test_prediction_and_cleanup(self):
        response = self.client.post("/predict", content=b"video")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(app.state.predictor.last_path.exists())

    def test_prediction_export_matches_response_and_expired_token_returns_404(self):
        import time
        result = self.client.post("/predict", content=b"video").json()
        url = result.pop("download_url")
        exported = self.client.get(url)
        self.assertEqual(exported.status_code, 200)
        self.assertEqual(exported.json(), result)
        self.assertIn("attachment", exported.headers["content-disposition"])
        token = url.removeprefix("/exports/").removesuffix(".json")
        app.state.exports[token]["created"] = time.monotonic() - 901
        self.assertEqual(self.client.get(url).status_code, 404)

    def test_export_memory_is_bounded(self):
        first = self.client.post("/predict", content=b"video").json()["download_url"]
        for _ in range(8):
            self.client.post("/predict", content=b"video")
        self.assertEqual(len(app.state.exports), 8)
        self.assertEqual(self.client.get(first).status_code, 404)

    def test_empty_and_invalid_upload(self):
        self.assertEqual(self.client.post("/predict", content=b"").status_code, 400)
        self.assertEqual(self.client.post("/predict", content=b"invalid").status_code, 422)
        self.assertFalse(app.state.predictor.last_path.exists())

    def test_size_limit(self):
        response = self.client.post("/predict", content=b"video", headers={"content-length": str(101 * 1024 ** 2)})
        self.assertEqual(response.status_code, 413)

    def test_invalid_stride(self):
        self.assertEqual(self.client.post("/predict?window=1&stride=2", content=b"video").status_code, 422)
        self.assertEqual(self.client.post("/predict?alpha=0", content=b"video").status_code, 422)

    def test_busy_worker(self):
        app.state.lock.acquire()
        try:
            self.assertEqual(self.client.post("/predict", content=b"video").status_code, 429)
        finally:
            app.state.lock.release()


if __name__ == "__main__":
    unittest.main()
