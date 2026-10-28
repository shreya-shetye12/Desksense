import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
class ModelTests(unittest.TestCase):
    def test_videomae_forward_backward_without_download(self):
        import torch
        from transformers import VideoMAEConfig
        from desksense.models import VideoClassifier
        cfg = VideoMAEConfig(image_size=32, patch_size=16, num_frames=4, tubelet_size=2,
                             hidden_size=24, num_hidden_layers=2, num_attention_heads=3,
                             intermediate_size=48, num_labels=2)
        spec = dict(model="videomae", hf_config=cfg.to_dict(), classes=["a", "b"])
        model = VideoClassifier(spec, pretrained=False)
        model.trainable("partial")
        x = torch.randn(2, 4, 3, 32, 32)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 2))
        logits.sum().backward()
        self.assertIsNotNone(model.network.classifier.weight.grad)
        self.assertIsNone(model.network.videomae.embeddings.patch_embeddings.projection.weight.grad)

    def test_videomae_without_pooling_norm(self):
        from transformers import VideoMAEConfig
        from desksense.models import VideoClassifier
        cfg = VideoMAEConfig(image_size=32, patch_size=16, num_frames=4, tubelet_size=2,
                             hidden_size=24, num_hidden_layers=2, num_attention_heads=3,
                             intermediate_size=48, num_labels=2, use_mean_pooling=False)
        model = VideoClassifier(dict(model="videomae", hf_config=cfg.to_dict(), classes=["a", "b"]),
                                pretrained=False)
        model.trainable("head")
        self.assertTrue(model.network.classifier.weight.requires_grad)
        self.assertIsNone(model.network.fc_norm)

    def test_frame_baseline_permutation_invariance(self):
        import torch
        from desksense.models import VideoClassifier
        model = VideoClassifier(dict(model="frame_mean", classes=["a", "b"]), pretrained=False).eval()
        x = torch.randn(1, 3, 3, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(model(x), model(x[:, [2, 0, 1]]), rtol=1e-5, atol=1e-5)

    def test_temporal_sampling_and_perturbations(self):
        import numpy as np
        from desksense.video import sample_indices, temporal_perturb
        np.testing.assert_array_equal(sample_indices(1, 4), [0, 0, 0, 0])
        frames = np.arange(16)[:, None]
        a = temporal_perturb(frames, "shuffle", "id", 42)
        np.testing.assert_array_equal(a, temporal_perturb(frames, "shuffle", "id", 42))
        np.testing.assert_array_equal(temporal_perturb(frames, "reverse")[:, 0], np.arange(15, -1, -1))
        np.testing.assert_array_equal(temporal_perturb(frames, "repeat")[:, 0], np.repeat(8, 16))


if __name__ == "__main__":
    unittest.main()
