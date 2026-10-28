import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from desksense.video import _metadata_cached, decode, metadata, sample_indices


class EstimatedCountCapture:
    """Container advertises a frame count larger than the actual decodable stream."""
    def __init__(self, *args):
        self.position = 0

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2
        return 12. if prop == cv2.CAP_PROP_FPS else 999.

    def grab(self):
        self.position += 1
        return self.position <= 4

    def read(self):
        self.position += 1
        return (True, np.full((16, 20, 3), self.position, dtype=np.uint8)) if self.position <= 4 else (False, None)

    def release(self):
        pass


class DecodeTests(unittest.TestCase):
    @patch('desksense.video.cv2.VideoCapture', EstimatedCountCapture)
    def test_actual_frames_instead_of_container_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'video.webm'
            path.write_bytes(b'fixture')
            _metadata_cached.cache_clear()
            count, fps = metadata(path)
            self.assertEqual(count, 4)
            self.assertEqual(fps, 12.)
            frames = decode(path, sample_indices(count, 16), 16)
            self.assertEqual(frames.shape, (16, 16, 16, 3))
            self.assertEqual(int(frames[0, 0, 0, 0]), 1)
            self.assertEqual(int(frames[-1, 0, 0, 0]), 4)


if __name__ == '__main__':
    unittest.main()
