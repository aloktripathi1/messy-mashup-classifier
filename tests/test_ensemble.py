"""
Tests for the ensemble logic in src/ensemble.py.

ensemble.py is an executable script (wandb, file I/O on import), so we
replicate and test its pure algorithmic pieces directly rather than importing
the module.
"""

import numpy as np
import pytest

N_SAMPLES = 20
N_CLASSES = 10
RNG = np.random.default_rng(42)


def make_probs(n=N_SAMPLES, k=N_CLASSES, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(k), size=n)
    return raw.astype(np.float32)


# ── weighted_ensemble ─────────────────────────────────────────────────────────
# Mirrors: ens_probs = w_cnn * cnn_probs + w_ast * ast_probs

def weighted_ensemble(probs_a, probs_b, w_a, w_b):
    return w_a * probs_a + w_b * probs_b


class TestWeightedEnsemble:
    def test_output_shape(self):
        a, b = make_probs(seed=0), make_probs(seed=1)
        out = weighted_ensemble(a, b, 0.3, 0.7)
        assert out.shape == (N_SAMPLES, N_CLASSES)

    def test_no_nan(self):
        a, b = make_probs(seed=0), make_probs(seed=1)
        out = weighted_ensemble(a, b, 0.3, 0.7)
        assert not np.any(np.isnan(out))

    def test_rows_sum_to_one(self):
        a, b = make_probs(seed=0), make_probs(seed=1)
        out = weighted_ensemble(a, b, 0.3, 0.7)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(N_SAMPLES), atol=1e-5)

    def test_equal_weights_is_average(self):
        a, b = make_probs(seed=0), make_probs(seed=1)
        out = weighted_ensemble(a, b, 0.5, 0.5)
        expected = (a + b) / 2
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_unit_weight_returns_single_model(self):
        a, b = make_probs(seed=0), make_probs(seed=1)
        out = weighted_ensemble(a, b, 1.0, 0.0)
        np.testing.assert_allclose(out, a, atol=1e-6)

    def test_argmax_is_valid_class(self):
        a, b = make_probs(seed=0), make_probs(seed=1)
        out = weighted_ensemble(a, b, 0.3, 0.7)
        preds = out.argmax(axis=1)
        assert preds.min() >= 0
        assert preds.max() < N_CLASSES


# ── temperature_scaling ───────────────────────────────────────────────────────
# Mirrors the pseudo-labeling sharpening block in ensemble.py

def temperature_scale(probs, temperature):
    logits = np.log(probs + 1e-10)
    scaled = logits / temperature
    exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp_s / exp_s.sum(axis=1, keepdims=True)


class TestTemperatureScaling:
    def test_output_shape(self):
        p = make_probs()
        out = temperature_scale(p, temperature=0.5)
        assert out.shape == p.shape

    def test_no_nan(self):
        p = make_probs()
        out = temperature_scale(p, temperature=0.5)
        assert not np.any(np.isnan(out))

    def test_rows_sum_to_one(self):
        p = make_probs()
        out = temperature_scale(p, temperature=0.5)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(N_SAMPLES), atol=1e-5)

    def test_sharpening_increases_max_confidence(self):
        p = make_probs()
        sharpened = temperature_scale(p, temperature=0.5)
        assert sharpened.max(axis=1).mean() > p.max(axis=1).mean()

    def test_argmax_preserved_after_sharpening(self):
        p = make_probs()
        sharpened = temperature_scale(p, temperature=0.5)
        np.testing.assert_array_equal(p.argmax(axis=1), sharpened.argmax(axis=1))

    def test_temperature_one_is_identity(self):
        p = make_probs()
        out = temperature_scale(p, temperature=1.0)
        np.testing.assert_allclose(out, p, atol=1e-4)


# ── pseudo_label_filtering ────────────────────────────────────────────────────
# Mirrors: pseudo_mask = max_confidence >= PSEUDO_CONFIDENCE

def filter_pseudo_labels(probs, threshold):
    max_conf = probs.max(axis=1)
    mask = max_conf >= threshold
    labels = probs[mask].argmax(axis=1)
    return mask, labels


class TestPseudoLabelFiltering:
    def test_mask_shape(self):
        p = make_probs()
        mask, _ = filter_pseudo_labels(p, threshold=0.3)
        assert mask.shape == (N_SAMPLES,)

    def test_labels_shape_matches_mask(self):
        p = make_probs()
        mask, labels = filter_pseudo_labels(p, threshold=0.3)
        assert labels.shape == (mask.sum(),)

    def test_no_nan_in_labels(self):
        p = make_probs()
        _, labels = filter_pseudo_labels(p, threshold=0.3)
        assert not np.any(np.isnan(labels.astype(float)))

    def test_high_threshold_gives_fewer_samples(self):
        p = make_probs()
        mask_low, _ = filter_pseudo_labels(p, threshold=0.15)
        mask_high, _ = filter_pseudo_labels(p, threshold=0.95)
        assert mask_high.sum() <= mask_low.sum()

    def test_zero_threshold_selects_all(self):
        p = make_probs()
        mask, _ = filter_pseudo_labels(p, threshold=0.0)
        assert mask.sum() == N_SAMPLES

    def test_threshold_above_one_selects_none(self):
        p = make_probs()
        mask, labels = filter_pseudo_labels(p, threshold=1.01)
        assert mask.sum() == 0
        assert labels.shape == (0,)

    def test_labels_are_valid_classes(self):
        p = make_probs()
        _, labels = filter_pseudo_labels(p, threshold=0.2)
        if labels.size > 0:
            assert labels.min() >= 0
            assert labels.max() < N_CLASSES


# ── three_model_ensemble ──────────────────────────────────────────────────────
# Mirrors: final3 = w_c * cnn_probs + w_a1 * ast_probs + w_a2 * ast_adapted

def three_model_ensemble(pa, pb, pc, wa, wb, wc):
    return wa * pa + wb * pb + wc * pc


class TestThreeModelEnsemble:
    def test_output_shape(self):
        a, b, c = make_probs(seed=0), make_probs(seed=1), make_probs(seed=2)
        out = three_model_ensemble(a, b, c, 0.2, 0.4, 0.4)
        assert out.shape == (N_SAMPLES, N_CLASSES)

    def test_no_nan(self):
        a, b, c = make_probs(seed=0), make_probs(seed=1), make_probs(seed=2)
        out = three_model_ensemble(a, b, c, 0.2, 0.4, 0.4)
        assert not np.any(np.isnan(out))

    def test_rows_sum_to_one(self):
        a, b, c = make_probs(seed=0), make_probs(seed=1), make_probs(seed=2)
        out = three_model_ensemble(a, b, c, 0.2, 0.4, 0.4)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(N_SAMPLES), atol=1e-5)

    def test_argmax_is_valid_class(self):
        a, b, c = make_probs(seed=0), make_probs(seed=1), make_probs(seed=2)
        out = three_model_ensemble(a, b, c, 0.3, 0.35, 0.35)
        preds = out.argmax(axis=1)
        assert preds.min() >= 0
        assert preds.max() < N_CLASSES
