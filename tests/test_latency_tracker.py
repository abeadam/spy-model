"""
Tests for src/evaluation/latency_tracker.py

Verifies latency measurement, violation flag logic, and that the wrapper
passes inputs and returns correctly.
"""

import time

import numpy as np
import pytest

from src.evaluation.latency_tracker import measure_inference_latency
from src.utils.config import DEFAULT_CONFIG, TrainingConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fast_predict_fn(sleep_seconds: float = 0.0):
    """
    Return a fake predict function that sleeps for `sleep_seconds` and returns
    a fixed prediction tuple.
    """
    def fake_predict(raw_input_window: np.ndarray):
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        predicted_changes   = np.array([0.001, 0.002, 0.003, 0.004, 0.005], dtype=np.float32)
        directional_signal  = "up"
        inner_latency_ms    = sleep_seconds * 1_000.0
        return predicted_changes, directional_signal, inner_latency_ms

    return fake_predict


def _make_raw_window() -> np.ndarray:
    return np.ones((DEFAULT_CONFIG.sequence_length, DEFAULT_CONFIG.n_features), dtype=np.float32)


# ── measure_inference_latency ─────────────────────────────────────────────────

class TestMeasureInferenceLatency:
    def test_returns_prediction_latency_and_flag(self):
        predict_fn = _make_fast_predict_fn()
        window     = _make_raw_window()
        result, outer_latency_ms, is_violation = measure_inference_latency(
            predict_fn, window, config=DEFAULT_CONFIG
        )
        assert result is not None
        assert isinstance(outer_latency_ms, float)
        assert isinstance(is_violation, bool)

    def test_prediction_output_is_passed_through_unchanged(self):
        predict_fn = _make_fast_predict_fn()
        window     = _make_raw_window()
        (predicted_changes, directional_signal, _), _, _ = measure_inference_latency(
            predict_fn, window, config=DEFAULT_CONFIG
        )
        expected_changes = np.array([0.001, 0.002, 0.003, 0.004, 0.005], dtype=np.float32)
        assert np.allclose(predicted_changes, expected_changes)
        assert directional_signal == "up"

    def test_latency_is_positive(self):
        predict_fn = _make_fast_predict_fn()
        window     = _make_raw_window()
        _, outer_latency_ms, _ = measure_inference_latency(
            predict_fn, window, config=DEFAULT_CONFIG
        )
        assert outer_latency_ms > 0.0

    def test_no_violation_for_fast_call(self):
        """A near-instant call must not trigger a violation."""
        predict_fn = _make_fast_predict_fn(sleep_seconds=0.0)
        window     = _make_raw_window()
        _, _, is_violation = measure_inference_latency(
            predict_fn, window, config=DEFAULT_CONFIG
        )
        assert is_violation is False

    def test_violation_triggered_above_threshold(self):
        """A call that takes > max_inference_latency_ms must set is_violation=True."""
        slow_config = TrainingConfig(max_inference_latency_ms=10.0)
        predict_fn  = _make_fast_predict_fn(sleep_seconds=0.050)  # 50ms > 10ms
        window      = _make_raw_window()
        _, outer_latency_ms, is_violation = measure_inference_latency(
            predict_fn, window, config=slow_config
        )
        assert outer_latency_ms > slow_config.max_inference_latency_ms
        assert is_violation is True

    def test_no_violation_just_below_threshold(self):
        """A call well below the threshold must not be flagged."""
        generous_config = TrainingConfig(max_inference_latency_ms=10_000.0)  # 10 seconds
        predict_fn      = _make_fast_predict_fn(sleep_seconds=0.0)
        window          = _make_raw_window()
        _, _, is_violation = measure_inference_latency(
            predict_fn, window, config=generous_config
        )
        assert is_violation is False

    def test_latency_captures_actual_sleep_duration(self):
        """Measured latency must be at least as long as the injected sleep."""
        sleep_ms   = 30
        predict_fn = _make_fast_predict_fn(sleep_seconds=sleep_ms / 1_000.0)
        window     = _make_raw_window()
        _, outer_latency_ms, _ = measure_inference_latency(
            predict_fn, window, config=DEFAULT_CONFIG
        )
        assert outer_latency_ms >= sleep_ms * 0.9  # Allow 10% timing slack.
