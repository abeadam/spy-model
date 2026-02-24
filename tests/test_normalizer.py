"""
Tests for src/data/normalizer.py

Covers:
  - normalize_window(): per-feature z-score output properties
  - denormalize_predictions(): inverse transform recovers original values
  - ValueError on NaN/Inf input
"""

import numpy as np
import pytest

from src.data.normalizer import denormalize_predictions, normalize_window


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_window(seq_len: int = 60, n_features: int = 9, seed: int = 0) -> np.ndarray:
    """Well-conditioned window with clearly non-zero variance per feature."""
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(490.0, 510.0, (seq_len, n_features)).astype(np.float32)


# ── normalize_window ──────────────────────────────────────────────────────────

class TestNormalizeWindow:
    def test_output_shape_matches_input(self):
        window     = _make_window(seq_len=60, n_features=9)
        normalized, _, _ = normalize_window(window)
        assert normalized.shape == window.shape

    def test_output_dtype_is_float32(self):
        normalized, means, stds = normalize_window(_make_window())
        assert normalized.dtype == np.float32
        assert means.dtype      == np.float32
        assert stds.dtype       == np.float32

    def test_per_feature_mean_is_approximately_zero(self):
        window     = _make_window()
        normalized, _, _ = normalize_window(window)
        per_feature_means = normalized.mean(axis=0)
        assert np.allclose(per_feature_means, 0.0, atol=1e-5)

    def test_per_feature_std_is_approximately_one(self):
        window     = _make_window()
        normalized, _, _ = normalize_window(window)
        per_feature_stds = normalized.std(axis=0)
        # The epsilon shift means std won't be exactly 1, but very close
        # for windows with non-trivial variance.
        assert np.allclose(per_feature_stds, 1.0, atol=0.01)

    def test_returned_means_match_input_column_means(self):
        window = _make_window()
        _, returned_means, _ = normalize_window(window)
        expected_means = window.mean(axis=0).astype(np.float32)
        assert np.allclose(returned_means, expected_means, rtol=1e-5)

    def test_returned_stds_match_input_column_stds(self):
        window = _make_window()
        _, _, returned_stds = normalize_window(window)
        expected_stds = window.std(axis=0).astype(np.float32)
        assert np.allclose(returned_stds, expected_stds, rtol=1e-5)

    def test_output_contains_no_nan_or_inf(self):
        normalized, _, _ = normalize_window(_make_window())
        assert np.isfinite(normalized).all()

    def test_raises_on_nan_in_input(self):
        window         = _make_window()
        window[5, 2]   = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            normalize_window(window)

    def test_raises_on_inf_in_input(self):
        window         = _make_window()
        window[10, 0]  = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            normalize_window(window)

    def test_flat_feature_does_not_raise(self):
        """A constant-valued feature has zero std; epsilon prevents zero division."""
        window        = _make_window()
        window[:, 3]  = 500.0   # flat feature
        normalized, _, _ = normalize_window(window)
        # The flat feature should normalize to approximately zero everywhere.
        assert np.isfinite(normalized).all()
        assert np.allclose(normalized[:, 3], 0.0, atol=1e-4)


# ── denormalize_predictions ───────────────────────────────────────────────────

class TestDenormalizePredictions:
    def test_inverse_transform_recovers_original_scale(self):
        """
        Normalize a window, then run denormalize on the spy_close column
        values from the normalized output.  The result must match the
        original spy_close values within float32 tolerance.
        """
        window = _make_window()
        normalized, means, stds = normalize_window(window)

        # Use the normalized spy_close values as synthetic "model predictions".
        normalized_spy = normalized[:, 0]
        recovered      = denormalize_predictions(normalized_spy, means, stds)
        original_spy   = window[:, 0].astype(np.float32)

        assert np.allclose(recovered, original_spy, rtol=1e-4)

    def test_output_dtype_is_float32(self):
        window = _make_window()
        _, means, stds = normalize_window(window)
        preds   = np.zeros(5, dtype=np.float32)
        result  = denormalize_predictions(preds, means, stds)
        assert result.dtype == np.float32

    def test_output_shape_matches_input(self):
        window = _make_window()
        _, means, stds = normalize_window(window)
        preds  = np.zeros(5, dtype=np.float32)
        result = denormalize_predictions(preds, means, stds)
        assert result.shape == preds.shape

    def test_zero_normalized_prediction_recovers_mean(self):
        """
        A prediction of 0 in normalized space should map to the mean of
        the spy_close column.
        """
        window = _make_window()
        _, means, stds = normalize_window(window)
        zero_pred  = np.zeros(1, dtype=np.float32)
        recovered  = denormalize_predictions(zero_pred, means, stds)
        spy_mean   = window[:, 0].mean()
        assert abs(float(recovered[0]) - spy_mean) < 0.01
