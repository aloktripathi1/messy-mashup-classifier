import sys
import types
import importlib

import numpy as np
import pytest

# Inject a stub config so augmentation.py can be imported without the real one.
_stub_config = types.ModuleType("config")
_stub_config.SR = 22050
_stub_config.TARGET_LEN = 22050 * 10  # 10 s
_stub_config.N_MELS = 128
_stub_config.N_FFT = 2048
_stub_config.HOP = 512
sys.modules.setdefault("config", _stub_config)

# Also stub librosa so load_wav can be imported (we won't call it in tests).
if "librosa" not in sys.modules:
    _stub_librosa = types.ModuleType("librosa")
    _stub_librosa.load = None
    _stub_feature = types.ModuleType("librosa.feature")
    _stub_librosa.feature = _stub_feature
    _stub_librosa.power_to_db = None
    sys.modules["librosa"] = _stub_librosa
    sys.modules["librosa.feature"] = _stub_feature

sys.path.insert(0, "src")
from augmentation import (
    add_noise_snr,
    apply_overdrive,
    apply_time_shift,
    spec_augment,
    apply_mixup,
)

SR = _stub_config.SR
TARGET_LEN = _stub_config.SR * 2  # 2-s mini signal for speed
N_MELS = 64
N_FRAMES = 32


def make_signal(length=TARGET_LEN):
    rng = np.random.default_rng(0)
    return rng.standard_normal(length).astype(np.float32)


def make_mel(n_mels=N_MELS, n_frames=N_FRAMES):
    rng = np.random.default_rng(1)
    return rng.standard_normal((n_mels, n_frames)).astype(np.float32)


# ── add_noise_snr ─────────────────────────────────────────────────────────────

class TestAddNoiseSNR:
    def test_output_shape(self):
        sig = make_signal()
        noise = make_signal()
        out = add_noise_snr(sig, noise, snr_db=20)
        assert out.shape == sig.shape

    def test_no_nan(self):
        sig = make_signal()
        noise = make_signal()
        out = add_noise_snr(sig, noise, snr_db=20)
        assert not np.any(np.isnan(out))

    def test_output_differs_from_input(self):
        sig = make_signal()
        noise = make_signal()
        out = add_noise_snr(sig, noise, snr_db=10)
        assert not np.allclose(out, sig)

    def test_silent_noise_returns_signal(self):
        sig = make_signal()
        noise = np.zeros_like(sig)
        out = add_noise_snr(sig, noise, snr_db=20)
        # Numerically close to original (noise ≈ 0 → scale ≈ 0)
        assert np.allclose(out, sig, atol=1e-3)


# ── apply_overdrive ───────────────────────────────────────────────────────────

class TestApplyOverdrive:
    def test_output_shape(self):
        sig = make_signal()
        out = apply_overdrive(sig)
        assert out.shape == sig.shape

    def test_no_nan(self):
        sig = make_signal()
        out = apply_overdrive(sig)
        assert not np.any(np.isnan(out))

    def test_clipped_to_unit(self):
        sig = make_signal()
        out = apply_overdrive(sig, gain=10.0)
        assert out.min() >= -1.0
        assert out.max() <= 1.0


# ── apply_time_shift ──────────────────────────────────────────────────────────

class TestApplyTimeShift:
    def test_output_shape(self):
        sig = make_signal()
        out = apply_time_shift(sig, max_shift=1000)
        assert out.shape == sig.shape

    def test_no_nan(self):
        sig = make_signal()
        out = apply_time_shift(sig, max_shift=1000)
        assert not np.any(np.isnan(out))

    def test_circular_shift_preserves_energy(self):
        sig = make_signal()
        out = apply_time_shift(sig, max_shift=1000)
        assert np.isclose(np.sum(sig ** 2), np.sum(out ** 2), rtol=1e-5)


# ── spec_augment ──────────────────────────────────────────────────────────────

class TestSpecAugment:
    def test_output_shape(self):
        mel = make_mel()
        out = spec_augment(mel)
        assert out.shape == mel.shape

    def test_no_nan(self):
        mel = make_mel()
        out = spec_augment(mel)
        assert not np.any(np.isnan(out))

    def test_input_not_mutated(self):
        mel = make_mel()
        original = mel.copy()
        spec_augment(mel)
        np.testing.assert_array_equal(mel, original)

    def test_some_values_zeroed(self):
        rng = np.random.default_rng(42)
        mel = rng.standard_normal((N_MELS, N_FRAMES)).astype(np.float32) + 10.0
        out = spec_augment(mel, num_freq_masks=2, freq_mask_size=5,
                           num_time_masks=2, time_mask_size=5)
        assert np.any(out == 0.0)


# ── apply_mixup ───────────────────────────────────────────────────────────────

class TestApplyMixup:
    def test_output_shape(self):
        mel1, mel2 = make_mel(), make_mel()
        mel_mix, lam, l1, l2 = apply_mixup(mel1, 0, mel2, 1)
        assert mel_mix.shape == mel1.shape

    def test_no_nan(self):
        mel1, mel2 = make_mel(), make_mel()
        mel_mix, lam, l1, l2 = apply_mixup(mel1, 0, mel2, 1)
        assert not np.any(np.isnan(mel_mix))

    def test_lambda_in_unit_interval(self):
        mel1, mel2 = make_mel(), make_mel()
        _, lam, _, _ = apply_mixup(mel1, 3, mel2, 7)
        assert 0.0 <= lam <= 1.0

    def test_labels_preserved(self):
        mel1, mel2 = make_mel(), make_mel()
        _, _, l1, l2 = apply_mixup(mel1, 3, mel2, 7)
        assert l1 == 3
        assert l2 == 7

    def test_blend_is_convex_combination(self):
        mel1 = np.ones((N_MELS, N_FRAMES), dtype=np.float32)
        mel2 = np.zeros((N_MELS, N_FRAMES), dtype=np.float32)
        mel_mix, lam, _, _ = apply_mixup(mel1, 0, mel2, 1)
        np.testing.assert_allclose(mel_mix, np.full_like(mel1, lam), rtol=1e-5)
