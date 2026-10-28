import unittest

import numpy as np

from desksense.metrics import (classification, choose_threshold, ema, fit_temperature,
                               group_bootstrap, paired_bootstrap, selective, softmax)


class MetricTests(unittest.TestCase):
    def test_known_confusion_and_f1(self):
        p = np.array([[.9, .1], [.8, .2], [.2, .8], [.1, .9]])
        m = classification([0, 1, 1, 1], p)
        self.assertEqual(m["confusion_matrix"], [[1, 0], [1, 2]])
        self.assertAlmostEqual(m["accuracy"], .75)
        self.assertAlmostEqual(m["macro_f1"], ((2 / 3) + (4 / 5)) / 2)
        self.assertEqual(m["support"], [1, 3])

    def test_perfect_bootstrap(self):
        y, p = [0, 1, 0, 1], np.eye(2)[[0, 1, 0, 1]]
        ci = group_bootstrap(y, p, ["a", "a", "b", "b"], 50)
        self.assertEqual(ci["accuracy"], [1., 1.])
        self.assertEqual(ci["macro_f1"], [1., 1.])

    def test_calibration_reduces_overconfidence_nll(self):
        logits = np.array([[8., 0.], [8., 0.], [0., 8.], [0., 8.]])
        y = [0, 1, 1, 0]
        t = fit_temperature(logits, y)
        self.assertGreater(t, 1.)
        self.assertLess(classification(y, softmax(logits / t))["nll"],
                        classification(y, softmax(logits))["nll"])
        np.testing.assert_array_equal(softmax(logits / t).argmax(1), logits.argmax(1))

    def test_confidence_one_is_in_last_bin(self):
        m = classification([0, 1], [[1., 0.], [0., 1.]])
        self.assertEqual(m["calibration_bins"][-1]["count"], 2)
        self.assertEqual(m["ece"], 0.)

    def test_ema_causality(self):
        p = np.array([[1., 0.], [0., 1.], [0., 1.]])
        np.testing.assert_allclose(ema(p, .5), [[1., 0.], [.5, .5], [.25, .75]])
        np.testing.assert_allclose(ema(p, .5)[:2], ema(p[:2], .5))
        with self.assertRaises(ValueError):
            ema(p, 0)

    def test_rejection_empty_and_target_unmet(self):
        p = np.array([[.6, .4], [.4, .6]])
        self.assertIsNone(selective([0, 1], p, .9)["accuracy"])
        self.assertTrue(choose_threshold([1, 0], p, .9)["target_unmet"])

    def test_paired_difference_zero(self):
        p = [[.8, .2], [.1, .9], [.7, .3], [.2, .8]]
        result = paired_bootstrap([0, 1, 0, 1], p, p, ["a", "b", "c", "d"], 20)
        self.assertEqual(result["ci95"], [0., 0.])

    def test_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            classification([0], [[float("nan"), .5]])
        with self.assertRaises(ValueError):
            classification([0], [[.5, .8]])

    def test_metrics_against_independent_sklearn_reference(self):
        from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score
        rng = np.random.default_rng(77)
        p = softmax(rng.normal(size=(100, 4)))
        y = rng.integers(0, 4, 100)
        m = classification(y, p)
        self.assertAlmostEqual(m["accuracy"], accuracy_score(y, p.argmax(1)))
        self.assertAlmostEqual(m["macro_f1"], f1_score(y, p.argmax(1), average="macro"))
        self.assertAlmostEqual(m["balanced_accuracy"], recall_score(y, p.argmax(1), average="macro"))
        self.assertAlmostEqual(m["nll"], log_loss(y, p))

    def test_stable_softmax(self):
        np.testing.assert_allclose(softmax([[10000., 10000.]]), [[.5, .5]])


if __name__ == "__main__":
    unittest.main()
