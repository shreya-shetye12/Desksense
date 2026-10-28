import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from desksense.common import read_json, write_json
from desksense.stream_eval import stream_evaluate


class FakePredictor:
    def __init__(self, *args):
        self.classes = ["a", "b"]
        self.source_fingerprints = []

    def sliding(self, *args):
        def p(end, raw, smooth):
            return dict(end=end, available_at=end, raw_probabilities=raw, probabilities=smooth,
                        raw_label=self.classes[raw.index(max(raw))], label=self.classes[smooth.index(max(smooth))])
        return dict(duration=4., predictions=[p(1., [.9, .1], [.9, .1]),
                                             p(2., [.9, .1], [.9, .1]),
                                             p(3., [.1, .9], [.6, .4]),
                                             p(4., [.1, .9], [.3, .7])])


class ContinuousTests(unittest.TestCase):
    def fixture(self, root):
        videos = []
        for split in ("val", "test"):
            path = root / (split + ".avi")
            path.write_bytes(split.encode())
            videos.append(dict(path=str(path), split=split, group=split,
                               segments=[dict(start=0., end=2., label="a"), dict(start=2., end=4., label="b")]))
        return videos

    def args(self, root, videos):
        write_json(root / "continuous.json", videos)
        return SimpleNamespace(run="fake", device="cpu", annotations=root / "continuous.json", split="test",
                               window=1., stride=1., alpha=.5, bootstrap=10, out=root / "results.json")

    @patch("desksense.stream_eval.Predictor", FakePredictor)
    def test_ground_truth_end_boundary_and_delay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.args(root, self.fixture(root))
            stream_evaluate(args)
            result = read_json(args.out)
            self.assertEqual(result["raw"]["accuracy"], 1.)
            self.assertEqual(result["smoothed"]["accuracy"], .75)
            self.assertEqual(result["raw_delays"]["median_detected_delay"], 1.)
            self.assertEqual(result["smooth_delays"]["median_detected_delay"], 2.)
            self.assertEqual(result["smooth_delays"]["missed"], 0)

    @patch("desksense.stream_eval.Predictor", FakePredictor)
    def test_cross_split_group_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = self.fixture(root)
            videos[1]["group"] = videos[0]["group"]
            with self.assertRaisesRegex(ValueError, "source-group leakage"):
                stream_evaluate(self.args(root, videos))

    @patch("desksense.stream_eval.Predictor", FakePredictor)
    def test_overlapping_annotations_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = self.fixture(root)
            videos[1]["segments"][1]["start"] = 1.
            with self.assertRaisesRegex(ValueError, "nonoverlapping"):
                stream_evaluate(self.args(root, videos))


if __name__ == "__main__":
    unittest.main()
