"""
Tests for src/model/spy_predictor.py

Covers: output shape, determinism, no NaN outputs, directional signal validity,
correct device placement, ValueError on bad input.
"""

import numpy as np
import pytest
import torch

from src.model.spy_predictor import SpyPredictor, _build_sinusoidal_position_encoding
from src.utils.config import DEFAULT_CONFIG, TrainingConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_raw_window(
    n_features: int = DEFAULT_CONFIG.n_features,
    sequence_length: int = DEFAULT_CONFIG.sequence_length,
    spy_close_value: float = 500.0,
) -> np.ndarray:
    """
    Build a synthetic raw (un-normalized) input window with realistic values.
    spy_close is in column 0 at spy_close_value.
    """
    rng = np.random.default_rng(seed=0)
    window = rng.uniform(0.5, 2.0, size=(sequence_length, n_features)).astype(np.float32)
    window[:, 0] = spy_close_value  # spy_close column
    return window


def _make_cpu_config() -> TrainingConfig:
    """Return a tiny config for fast CPU-only tests."""
    return TrainingConfig(
        d_model=32,
        n_heads=2,
        n_encoder_layers=1,
        ffn_dim=64,
        n_features=DEFAULT_CONFIG.n_features,
        sequence_length=DEFAULT_CONFIG.sequence_length,
        prediction_steps=DEFAULT_CONFIG.prediction_steps,
    )


# ── Sinusoidal positional encoding ────────────────────────────────────────────

class TestBuildSinusoidalPositionEncoding:
    def test_output_shape(self):
        seq_len  = DEFAULT_CONFIG.sequence_length
        encoding = _build_sinusoidal_position_encoding(sequence_length=seq_len, d_model=128)
        assert encoding.shape == (1, seq_len, 128)

    def test_all_values_finite(self):
        encoding = _build_sinusoidal_position_encoding(
            sequence_length=DEFAULT_CONFIG.sequence_length, d_model=128
        )
        assert torch.isfinite(encoding).all()

    def test_different_positions_differ(self):
        seq_len  = DEFAULT_CONFIG.sequence_length
        encoding = _build_sinusoidal_position_encoding(sequence_length=seq_len, d_model=128)
        # No two rows should be identical (each position has a unique encoding).
        for i in range(seq_len - 1):
            assert not torch.allclose(encoding[0, i], encoding[0, i + 1])


# ── SpyPredictor forward pass ─────────────────────────────────────────────────

class TestSpyPredictorForward:
    def test_output_shape_for_single_sample(self):
        config = _make_cpu_config()
        model  = SpyPredictor(config=config, device=torch.device("cpu"))
        x      = torch.randn(1, config.sequence_length, config.n_features)
        output = model(x)
        assert output.shape == (1, config.prediction_steps)

    def test_output_shape_for_batch(self):
        config     = _make_cpu_config()
        model      = SpyPredictor(config=config, device=torch.device("cpu"))
        batch_size = 8
        x          = torch.randn(batch_size, config.sequence_length, config.n_features)
        output     = model(x)
        assert output.shape == (batch_size, config.prediction_steps)

    def test_no_nan_in_output(self):
        config = _make_cpu_config()
        model  = SpyPredictor(config=config, device=torch.device("cpu"))
        x      = torch.randn(4, config.sequence_length, config.n_features)
        output = model(x)
        assert not torch.isnan(output).any()

    def test_output_is_deterministic_in_eval_mode(self):
        config = _make_cpu_config()
        model  = SpyPredictor(config=config, device=torch.device("cpu"))
        model.eval()
        x       = torch.randn(1, config.sequence_length, config.n_features)
        output1 = model(x)
        output2 = model(x)
        assert torch.allclose(output1, output2)


# ── SpyPredictor.predict() ────────────────────────────────────────────────────

class TestSpyPredictorPredict:
    def test_predicted_changes_shape(self):
        config = _make_cpu_config()
        model  = SpyPredictor(config=config, device=torch.device("cpu"))
        window = _make_raw_window()
        predicted_changes, _, _ = model.predict(window)
        assert predicted_changes.shape == (config.prediction_steps,)

    def test_predicted_changes_are_float32(self):
        config = _make_cpu_config()
        model  = SpyPredictor(config=config, device=torch.device("cpu"))
        window = _make_raw_window()
        predicted_changes, _, _ = model.predict(window)
        assert predicted_changes.dtype == np.float32

    def test_no_nan_in_predicted_changes(self):
        config = _make_cpu_config()
        model  = SpyPredictor(config=config, device=torch.device("cpu"))
        window = _make_raw_window()
        predicted_changes, _, _ = model.predict(window)
        assert not np.isnan(predicted_changes).any()

    def test_directional_signal_is_up_or_down(self):
        config = _make_cpu_config()
        model  = SpyPredictor(config=config, device=torch.device("cpu"))
        window = _make_raw_window()
        _, directional_signal, _ = model.predict(window)
        assert directional_signal in {"up", "down"}

    def test_predict_is_deterministic_in_eval_mode(self):
        config = _make_cpu_config()
        model  = SpyPredictor(config=config, device=torch.device("cpu"))
        model.eval()
        window = _make_raw_window()
        changes1, signal1, _ = model.predict(window)
        changes2, signal2, _ = model.predict(window)
        assert np.allclose(changes1, changes2)
        assert signal1 == signal2

    def test_latency_ms_is_positive(self):
        config = _make_cpu_config()
        model  = SpyPredictor(config=config, device=torch.device("cpu"))
        window = _make_raw_window()
        _, _, latency_ms = model.predict(window)
        assert latency_ms > 0.0

    def test_raises_on_wrong_input_shape(self):
        config     = _make_cpu_config()
        model      = SpyPredictor(config=config, device=torch.device("cpu"))
        bad_window = np.ones((30, config.n_features), dtype=np.float32)  # wrong seq len
        with pytest.raises(ValueError, match="shape"):
            model.predict(bad_window)

    def test_raises_on_nan_in_input(self):
        config        = _make_cpu_config()
        model         = SpyPredictor(config=config, device=torch.device("cpu"))
        window_with_nan = _make_raw_window()
        window_with_nan[5, 2] = float("nan")
        with pytest.raises(ValueError):
            model.predict(window_with_nan)

    def test_directional_signal_is_up_when_last_change_is_positive(self):
        """Force model output so last predicted percent change > 0.0."""
        config = _make_cpu_config()
        model  = SpyPredictor(config=config, device=torch.device("cpu"))

        # Bias the output projection so all outputs are strongly positive.
        with torch.no_grad():
            model.output_projection.weight.fill_(0.0)
            model.output_projection.bias.fill_(1.0)

        window = _make_raw_window()
        _, directional_signal, _ = model.predict(window)
        assert directional_signal == "up"

    def test_model_is_on_correct_device(self):
        config = _make_cpu_config()
        model  = SpyPredictor(config=config, device=torch.device("cpu"))
        device = next(model.parameters()).device
        assert device == torch.device("cpu")

    def test_parameter_count_is_reasonable(self):
        """
        Default config (d=128, h=4, L=3, ffn=512) produces ~595K parameters.
        The FFN (2 × 128×512 per block × 3 blocks) dominates the count.
        Confirm the model is in the expected order of magnitude.
        """
        model = SpyPredictor(config=DEFAULT_CONFIG, device=torch.device("cpu"))
        param_count = model.parameter_count()
        assert 400_000 < param_count < 1_000_000
