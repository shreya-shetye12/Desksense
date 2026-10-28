import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from desksense.api import app


class DashboardPredictor:
    def __init__(self, *args):
        self.classes = ['Picking something up', 'Opening something']
        self.device = 'cpu'
        self.spec = {'model': 'videomae'}


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'DESKSENSE_RUN': 'fake'})
        self.predictor = patch('desksense.api.Predictor', DashboardPredictor)
        self.env.start()
        self.predictor.start()
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.predictor.stop()
        self.env.stop()

    def test_dashboard_and_packaged_assets_are_served(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.headers['content-type'])
        self.assertIn('id="confidenceChart"', response.text)
        self.assertIn('id="fileInput"', response.text)
        for asset in ('app.js', 'styles.css'):
            result = self.client.get('/static/' + asset)
            self.assertEqual(result.status_code, 200)
            self.assertGreater(len(result.content), 100)

    def test_health_identifies_loaded_model(self):
        result = self.client.get('/health').json()
        self.assertEqual(result['model_name'], 'VideoMAE')
        self.assertEqual(result['device'], 'cpu')
        self.assertEqual(len(result['classes']), 2)

    def test_sample_is_opt_in_and_restricted_to_configured_file(self):
        with patch.dict(os.environ, {'DESKSENSE_SAMPLE_VIDEO': ''}):
            self.assertEqual(self.client.get('/sample-video').status_code, 404)
            self.assertFalse(self.client.get('/health').json()['sample_available'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.webm'
            path.write_bytes(b'configured-sample')
            with patch.dict(os.environ, {'DESKSENSE_SAMPLE_VIDEO': str(path)}):
                self.assertTrue(self.client.get('/health').json()['sample_available'])
                self.assertEqual(self.client.get('/sample-video?path=/etc/passwd').content, b'configured-sample')


if __name__ == '__main__':
    unittest.main()
