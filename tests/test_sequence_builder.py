"""
Tests for src/data/sequence_builder.py

Verifies no cross-day windows, correct shapes, no NaN outputs, and
correct sequence count arithmetic.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.sequence_builder import (
    FEATURE_COLUMN_NAMES,
    build_all_sequences,
    build_sequences_for_day,
)
from src.utils.config import DEFAULT_CONFIG, TrainingConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_feature_frame(
    n_bars: int,
    price: float = 500.0,
    volume: int = 1_000,
) -> pd.DataFrame:
    """
    Build a synthetic feature DataFrame with all 26 required columns and no NaN
    values — simulating a post-indicator-computation, post-warmup-drop frame.
    """
    timestamps = pd.RangeIndex(n_bars)
    prices     = np.linspace(price, price + n_bars * 0.01, n_bars, dtype=np.float32)
    data = {
        # Core features
        "spy_close":           prices,
        "vix_close":           np.full(n_bars, 18.0,   dtype=np.float32),
        "ema_5":               prices * 0.9999,
        "ema_9":               prices * 0.9998,
        "ema_12":              prices * 0.9997,
        # RSI (all periods)
        "rsi_7":               np.full(n_bars, 55.0,   dtype=np.float32),
        "rsi_14":              np.full(n_bars, 50.0,   dtype=np.float32),
        "rsi_28":              np.full(n_bars, 50.0,   dtype=np.float32),
        # Momentum / VWAP
        "vwap_deviation":      np.full(n_bars, 0.01,   dtype=np.float32),
        "price_momentum_5":    np.full(n_bars, 0.02,   dtype=np.float32),
        "volume_momentum_5":   np.full(n_bars, 0.0,    dtype=np.float32),
        # Bollinger Bands
        "bb_pct_b_7":          np.full(n_bars, 0.5,    dtype=np.float32),
        "bb_pct_b_14":         np.full(n_bars, 0.5,    dtype=np.float32),
        "bb_pct_b_28":         np.full(n_bars, 0.5,    dtype=np.float32),
        "bb_width_7":          np.full(n_bars, 0.02,   dtype=np.float32),
        "bb_width_14":         np.full(n_bars, 0.02,   dtype=np.float32),
        "bb_width_28":         np.full(n_bars, 0.02,   dtype=np.float32),
        # ATR
        "atr_7":               np.full(n_bars, 0.01,   dtype=np.float32),
        "atr_14":              np.full(n_bars, 0.01,   dtype=np.float32),
        "atr_28":              np.full(n_bars, 0.01,   dtype=np.float32),
        # MFI
        "mfi_7":               np.full(n_bars, 50.0,   dtype=np.float32),
        "mfi_14":              np.full(n_bars, 50.0,   dtype=np.float32),
        "mfi_28":              np.full(n_bars, 50.0,   dtype=np.float32),
        # Realized volatility
        "realized_vol_7":      np.full(n_bars, 0.001,  dtype=np.float32),
        "realized_vol_14":     np.full(n_bars, 0.001,  dtype=np.float32),
        "realized_vol_28":     np.full(n_bars, 0.001,  dtype=np.float32),
        # Extra columns that indicator_computer adds but sequence_builder ignores.
        "spy_open":            prices,
        "spy_high":            prices + 0.1,
        "spy_low":             prices - 0.1,
        "spy_volume":          np.full(n_bars, volume, dtype=np.int64),
        "vix_open":            np.full(n_bars, 18.0,   dtype=np.float32),
    }
    return pd.DataFrame(data, index=pd.Index(timestamps, name="unix_timestamp"))


# ── FEATURE_COLUMN_NAMES contract ─────────────────────────────────────────────

class TestFeatureColumnNames:
    def test_has_exactly_26_features(self):
        assert len(FEATURE_COLUMN_NAMES) == DEFAULT_CONFIG.n_features
        assert len(FEATURE_COLUMN_NAMES) == 26

    def test_spy_close_is_first(self):
        assert FEATURE_COLUMN_NAMES[0] == "spy_close"

    def test_contains_all_multi_period_indicators(self):
        """All 3 periods for each multi-period indicator must be present."""
        for period in (7, 14, 28):
            assert f"rsi_{period}"          in FEATURE_COLUMN_NAMES
            assert f"bb_pct_b_{period}"     in FEATURE_COLUMN_NAMES
            assert f"bb_width_{period}"     in FEATURE_COLUMN_NAMES
            assert f"atr_{period}"          in FEATURE_COLUMN_NAMES
            assert f"mfi_{period}"          in FEATURE_COLUMN_NAMES
            assert f"realized_vol_{period}" in FEATURE_COLUMN_NAMES


# ── build_sequences_for_day ───────────────────────────────────────────────────

class TestBuildSequencesForDay:
    def test_output_shapes_are_correct(self):
        config    = DEFAULT_CONFIG
        n_bars    = config.sequence_length + config.prediction_steps + 10
        frame     = _make_feature_frame(n_bars)
        sequences, targets = build_sequences_for_day(frame, date(2024, 1, 2), config)

        assert sequences.shape[1] == config.sequence_length
        assert sequences.shape[2] == config.n_features
        assert targets.shape[1]   == config.prediction_steps

    def test_sequence_count_matches_sliding_window_arithmetic(self):
        config    = DEFAULT_CONFIG
        n_bars    = 200
        frame     = _make_feature_frame(n_bars)
        sequences, targets = build_sequences_for_day(frame, date(2024, 1, 2), config)

        expected_count = n_bars - config.sequence_length - config.prediction_steps + 1
        assert len(sequences) == expected_count
        assert len(targets)   == expected_count

    def test_no_nan_in_sequences(self):
        config    = DEFAULT_CONFIG
        frame     = _make_feature_frame(200)
        sequences, _ = build_sequences_for_day(frame, date(2024, 1, 2), config)
        assert not np.isnan(sequences).any()

    def test_no_nan_in_targets(self):
        config = DEFAULT_CONFIG
        frame  = _make_feature_frame(200)
        _, targets = build_sequences_for_day(frame, date(2024, 1, 2), config)
        assert not np.isnan(targets).any()

    def test_targets_are_percent_changes(self):
        """Targets must be percent-change returns, not absolute dollar prices."""
        config = DEFAULT_CONFIG
        frame  = _make_feature_frame(200, price=500.0)
        _, targets = build_sequences_for_day(frame, date(2024, 1, 2), config)
        # Percent changes from a ~500 base with 0.01/bar drift are tiny fractions.
        # Absolute prices would be ~500; percent changes are < 1.0.
        assert np.abs(targets).max() < 1.0

    def test_returns_empty_arrays_when_insufficient_bars(self):
        config    = DEFAULT_CONFIG
        too_short = config.sequence_length + config.prediction_steps - 1
        frame     = _make_feature_frame(too_short)
        sequences, targets = build_sequences_for_day(frame, date(2024, 1, 2), config)

        assert sequences.shape[0] == 0
        assert targets.shape[0]   == 0

    def test_sequences_are_float32(self):
        config = DEFAULT_CONFIG
        frame  = _make_feature_frame(200)
        sequences, _ = build_sequences_for_day(frame, date(2024, 1, 2), config)
        assert sequences.dtype == np.float32

    def test_targets_are_float32(self):
        config = DEFAULT_CONFIG
        frame  = _make_feature_frame(200)
        _, targets = build_sequences_for_day(frame, date(2024, 1, 2), config)
        assert targets.dtype == np.float32

    def test_dropna_removes_warmup_rows_before_slicing(self):
        """
        Rows with NaN (indicator warmup) must be removed before windows are built,
        so no window ever contains a NaN feature value.
        """
        config = DEFAULT_CONFIG
        frame  = _make_feature_frame(300)
        # Inject NaN into the first 40 rows of ema_12 (simulating warmup).
        frame = frame.copy()
        frame.loc[frame.index[:40], "ema_12"] = np.nan

        sequences, _ = build_sequences_for_day(frame, date(2024, 1, 2), config)
        assert not np.isnan(sequences).any()


# ── vol_normalized_log_prevday transform ──────────────────────────────────────

class TestVolNormalizedLogPrevdayTransform:
    def test_returns_non_nan_targets_when_prev_day_vol_provided(self):
        """When prev_day_realized_vol is given, sequences and targets are valid."""
        config = TrainingConfig(target_transform="vol_normalized_log_prevday")
        frame  = _make_feature_frame(200)
        sequences, targets = build_sequences_for_day(
            frame, date(2024, 1, 2), config, prev_day_realized_vol=0.001
        )
        assert sequences.shape[0] > 0
        assert not np.isnan(targets).any()
        assert not np.isnan(sequences).any()

    def test_returns_empty_arrays_when_prev_day_vol_is_none(self):
        """First trading day has no prior vol — must return empty arrays silently."""
        config = TrainingConfig(target_transform="vol_normalized_log_prevday")
        frame  = _make_feature_frame(200)
        sequences, targets = build_sequences_for_day(
            frame, date(2024, 1, 2), config, prev_day_realized_vol=None
        )
        assert sequences.shape[0] == 0
        assert targets.shape[0]   == 0

    def test_targets_are_scaled_by_prev_day_vol(self):
        """
        Targets for vol_normalized_log_prevday should scale inversely with the vol
        denominator: smaller vol → larger normalized targets.
        """
        frame  = _make_feature_frame(200)
        config_low_vol  = TrainingConfig(target_transform="vol_normalized_log_prevday")
        config_high_vol = TrainingConfig(target_transform="vol_normalized_log_prevday")

        _, targets_low_vol = build_sequences_for_day(
            frame, date(2024, 1, 2), config_low_vol, prev_day_realized_vol=0.0001
        )
        _, targets_high_vol = build_sequences_for_day(
            frame, date(2024, 1, 2), config_high_vol, prev_day_realized_vol=0.01
        )
        # Lower vol denominator → larger normalized target magnitudes.
        assert np.abs(targets_low_vol).mean() > np.abs(targets_high_vol).mean()

    def test_build_all_sequences_skips_first_day_for_prevday_transform(self):
        """With 2 days and prevday transform, day 1 is skipped (no prior vol)."""
        config = TrainingConfig(target_transform="vol_normalized_log_prevday")
        day_frames = {
            date(2024, 1, 2): _make_feature_frame(200, price=500.0),
            date(2024, 1, 3): _make_feature_frame(200, price=500.5),
        }
        sequences, _, sequence_dates = build_all_sequences(day_frames, config)
        # Only day 2 should produce sequences (day 1 has no prior vol).
        assert all(d == date(2024, 1, 3) for d in sequence_dates)

    def test_build_all_sequences_prevday_raises_if_only_one_day(self):
        """With a single day and prevday transform, RuntimeError because 0 sequences total."""
        config = TrainingConfig(target_transform="vol_normalized_log_prevday")
        day_frames = {
            date(2024, 1, 2): _make_feature_frame(200, price=500.0),
        }
        with pytest.raises(RuntimeError):
            build_all_sequences(day_frames, config)


# ── build_all_sequences ───────────────────────────────────────────────────────

class TestBuildAllSequences:
    def test_sequences_across_multiple_days_do_not_mix(self):
        """
        Each day's windows are built independently, so a window from day 1
        can never contain bars from day 2.
        """
        config = DEFAULT_CONFIG
        day_frames = {
            date(2024, 1, 2): _make_feature_frame(200, price=500.0),
            date(2024, 1, 3): _make_feature_frame(200, price=600.0),
        }
        sequences, targets, sequence_dates = build_all_sequences(day_frames, config)

        day1_count = 200 - config.sequence_length - config.prediction_steps + 1
        day2_count = 200 - config.sequence_length - config.prediction_steps + 1
        assert len(sequences) == day1_count + day2_count

    def test_sequence_dates_length_matches_sequences(self):
        config = DEFAULT_CONFIG
        day_frames = {
            date(2024, 1, 2): _make_feature_frame(200),
            date(2024, 1, 3): _make_feature_frame(200),
        }
        sequences, _, sequence_dates = build_all_sequences(day_frames, config)
        assert len(sequence_dates) == len(sequences)

    def test_sequence_dates_are_chronological(self):
        config = DEFAULT_CONFIG
        day_frames = {
            date(2024, 1, 4): _make_feature_frame(200),
            date(2024, 1, 2): _make_feature_frame(200),
            date(2024, 1, 3): _make_feature_frame(200),
        }
        _, _, sequence_dates = build_all_sequences(day_frames, config)
        assert sequence_dates == sorted(sequence_dates)

    def test_raises_when_no_sequences_produced(self):
        config = DEFAULT_CONFIG
        # Every day too short → no sequences.
        too_short = config.sequence_length + config.prediction_steps - 1
        day_frames = {
            date(2024, 1, 2): _make_feature_frame(too_short),
        }
        with pytest.raises(RuntimeError):
            build_all_sequences(day_frames, config)
