"""
Tests for src/data/indicator_computer.py

Uses synthetic price/volume series — no real bar files required.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.indicator_computer import (
    _compute_atr_series,
    _compute_bollinger_bands_series,
    _compute_ema_series,
    _compute_mfi_series,
    _compute_momentum_series,
    _compute_realized_vol_series,
    _compute_rsi_series,
    _compute_vwap_deviation_series,
    compute_indicators,
)
from src.utils.config import DEFAULT_CONFIG


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_aligned_frame(
    n_bars: int = 200,
    price: float = 500.0,
    volume: int = 1_000,
    price_trend: float = 0.0,
) -> pd.DataFrame:
    """
    Build a synthetic aligned daily bar DataFrame.

    price_trend: added cumulatively per bar so price drifts upward/downward.
    """
    timestamps = pd.RangeIndex(start=1_700_000_000, stop=1_700_000_000 + n_bars * 5, step=5)
    prices = np.array([price + price_trend * i for i in range(n_bars)], dtype=np.float32)
    return pd.DataFrame(
        {
            "spy_open":   prices,
            "spy_high":   prices + 0.10,
            "spy_low":    prices - 0.10,
            "spy_close":  prices,
            "spy_volume": np.full(n_bars, volume, dtype=np.int64),
            "vix_close":  np.full(n_bars, 18.0, dtype=np.float32),
        },
        index=pd.Index(timestamps, name="unix_timestamp"),
    )


# ── EMA ───────────────────────────────────────────────────────────────────────

class TestComputeEmaSeries:
    def test_ema_converges_to_flat_price(self):
        """After warmup, EMA of a constant series equals that constant."""
        flat_price   = pd.Series(np.full(100, 500.0))
        ema_5_series = _compute_ema_series(flat_price, period=5)
        valid_values = ema_5_series.dropna()
        assert np.allclose(valid_values.values, 500.0, rtol=1e-4)

    def test_warmup_rows_are_nan(self):
        price  = pd.Series(np.full(100, 500.0))
        period = 5
        result = _compute_ema_series(price, period=period)
        warmup = period * 3 - 1  # first `warmup` rows should be NaN
        assert result.iloc[:warmup].isna().all()
        assert result.iloc[warmup:].notna().all()

    def test_ema_tracks_rising_price(self):
        """EMA must be less than current price when price is trending up."""
        rising_price = pd.Series(np.arange(200, dtype=float))
        ema          = _compute_ema_series(rising_price, period=5)
        valid        = ema.dropna()
        # EMA lags price on an uptrend, so it should be below the current price.
        assert (rising_price.loc[valid.index] > valid).all()


# ── RSI ───────────────────────────────────────────────────────────────────────

class TestComputeRsiSeries:
    def test_rsi_stays_within_bounds(self):
        rng   = np.random.default_rng(seed=42)
        price = pd.Series(500 + rng.normal(0, 1, 300).cumsum())
        rsi   = _compute_rsi_series(price, period=7)
        valid = rsi.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_rsi_is_100_for_strictly_rising_price(self):
        """When price only goes up (no losses), RSI should reach 100 after warmup."""
        strictly_rising = pd.Series(np.arange(1, 101, dtype=float))
        rsi             = _compute_rsi_series(strictly_rising, period=7)
        valid           = rsi.dropna()
        assert np.allclose(valid.values, 100.0, atol=1e-4)

    def test_rsi_is_0_for_strictly_falling_price(self):
        """When price only falls (no gains), RSI should reach 0 after warmup."""
        strictly_falling = pd.Series(np.arange(100, 0, -1, dtype=float))
        rsi              = _compute_rsi_series(strictly_falling, period=7)
        valid            = rsi.dropna()
        assert np.allclose(valid.values, 0.0, atol=1e-4)

    def test_rsi_is_50_for_flat_price(self):
        """Flat price has zero gains and zero losses; RSI is defined as 50."""
        flat_price = pd.Series(np.full(100, 500.0))
        rsi        = _compute_rsi_series(flat_price, period=7)
        valid      = rsi.dropna()
        assert np.allclose(valid.values, 50.0, atol=1e-4)

    def test_warmup_rows_are_nan(self):
        price  = pd.Series(np.arange(100, dtype=float))
        period = 7
        rsi    = _compute_rsi_series(price, period=period)
        warmup = period * 3 - 1
        assert rsi.iloc[:warmup].isna().all()
        assert rsi.iloc[warmup:].notna().all()


# ── RSI multi-period ──────────────────────────────────────────────────────────

class TestComputeRsiMultiPeriod:
    def test_rsi_14_stays_within_bounds(self):
        rng   = np.random.default_rng(seed=42)
        price = pd.Series(500 + rng.normal(0, 1, 400).cumsum())
        rsi   = _compute_rsi_series(price, period=14)
        valid = rsi.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_rsi_28_stays_within_bounds(self):
        rng   = np.random.default_rng(seed=7)
        price = pd.Series(500 + rng.normal(0, 1, 400).cumsum())
        rsi   = _compute_rsi_series(price, period=28)
        valid = rsi.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_rsi_14_warmup_rows_are_nan(self):
        price  = pd.Series(np.arange(200, dtype=float))
        period = 14
        rsi    = _compute_rsi_series(price, period=period)
        warmup = period * 3 - 1  # 41
        assert rsi.iloc[:warmup].isna().all()
        assert rsi.iloc[warmup:].notna().all()

    def test_rsi_28_warmup_rows_are_nan(self):
        price  = pd.Series(np.arange(300, dtype=float))
        period = 28
        rsi    = _compute_rsi_series(price, period=period)
        warmup = period * 3 - 1  # 83
        assert rsi.iloc[:warmup].isna().all()
        assert rsi.iloc[warmup:].notna().all()


# ── VWAP Deviation ────────────────────────────────────────────────────────────

class TestComputeVwapDeviationSeries:
    def test_deviation_is_zero_when_price_equals_vwap(self):
        """Constant price and constant volume → VWAP = price → deviation = 0."""
        n       = 200
        price   = pd.Series(np.full(n, 500.0))
        volume  = pd.Series(np.full(n, 1_000))
        result  = _compute_vwap_deviation_series(price, price, price, volume, vwap_window=60)
        assert np.allclose(result.values, 0.0, atol=1e-4)

    def test_deviation_sign_when_price_above_vwap(self):
        """If current price jumps above its historical average, deviation > 0."""
        n       = 200
        price   = pd.Series(np.full(n, 500.0))
        volume  = pd.Series(np.full(n, 1_000))
        # Spike the last bar's price above the rolling average.
        price_spiked          = price.copy()
        price_spiked.iloc[-1] = 520.0
        result = _compute_vwap_deviation_series(
            price_spiked, price_spiked, price_spiked, volume, vwap_window=60
        )
        assert result.iloc[-1] > 0.0

    def test_handles_zero_volume_without_error(self):
        """All-zero volume (e.g. VIX series) must not raise or produce Inf."""
        n      = 200
        price  = pd.Series(np.full(n, 18.0))
        volume = pd.Series(np.zeros(n))
        result = _compute_vwap_deviation_series(price, price, price, volume, vwap_window=60)
        assert np.isfinite(result).all()


# ── Momentum ──────────────────────────────────────────────────────────────────

class TestComputeMomentumSeries:
    def test_positive_momentum_for_rising_price(self):
        """Linearly rising price → positive 5-bar momentum throughout."""
        price    = pd.Series(np.arange(100, dtype=float))
        momentum = _compute_momentum_series(price, period=5)
        valid    = momentum.dropna()
        assert (valid > 0.0).all()

    def test_negative_momentum_for_falling_price(self):
        falling = pd.Series(np.arange(100, dtype=float)[::-1].copy())
        momentum = _compute_momentum_series(falling, period=5)
        valid    = momentum.dropna()
        assert (valid < 0.0).all()

    def test_first_period_rows_are_nan(self):
        # Start from 1 to avoid a zero denominator at the lagged position.
        price    = pd.Series(np.arange(1, 51, dtype=float))
        momentum = _compute_momentum_series(price, period=5)
        assert momentum.iloc[:5].isna().all()
        assert momentum.iloc[5:].notna().all()


# ── Bollinger Bands ───────────────────────────────────────────────────────────

class TestComputeBollingerBands:
    def test_bb_pct_b_within_bounds_after_warmup(self):
        """bb_pct_b must be clipped to [0, 1] after the warmup period."""
        rng   = np.random.default_rng(seed=42)
        price = pd.Series(500 + rng.normal(0, 1, 300).cumsum())
        bb_pct_b, _ = _compute_bollinger_bands_series(price, period=7)
        valid = bb_pct_b.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_bb_width_is_non_negative_after_warmup(self):
        rng   = np.random.default_rng(seed=0)
        price = pd.Series(500 + rng.normal(0, 1, 200).cumsum())
        _, bb_width = _compute_bollinger_bands_series(price, period=14)
        valid = bb_width.dropna()
        assert (valid >= 0.0).all()

    def test_warmup_rows_are_nan(self):
        price  = pd.Series(np.arange(1, 201, dtype=float))
        period = 14
        bb_pct_b, bb_width = _compute_bollinger_bands_series(price, period=period)
        warmup = period - 1
        assert bb_pct_b.iloc[:warmup].isna().all()
        assert bb_width.iloc[:warmup].isna().all()

    def test_flat_price_has_zero_width(self):
        """Flat price has zero std → zero band width → bb_pct_b is NaN (0/0)."""
        price = pd.Series(np.full(100, 500.0))
        bb_pct_b, bb_width = _compute_bollinger_bands_series(price, period=7)
        # Width should be 0 for flat price (upper == lower == mean → range = 0).
        valid_width = bb_width.dropna()
        assert np.allclose(valid_width.values, 0.0, atol=1e-6)

    def test_rising_price_produces_positive_bb_pct_b(self):
        """On a consistent uptrend, close approaches upper band → bb_pct_b near 1."""
        rising_price = pd.Series(np.linspace(490, 510, 200))
        bb_pct_b, _ = _compute_bollinger_bands_series(rising_price, period=7)
        valid = bb_pct_b.dropna()
        # Not strictly 1.0, but should be above 0.5 on a trending series.
        assert valid.mean() > 0.5


# ── ATR ───────────────────────────────────────────────────────────────────────

class TestComputeAtr:
    def test_atr_is_non_negative_after_warmup(self):
        rng   = np.random.default_rng(seed=42)
        prices = 500 + rng.normal(0, 1, 200).cumsum()
        price_series = pd.Series(prices)
        high   = price_series + 0.5
        low    = price_series - 0.5
        atr = _compute_atr_series(high, low, price_series, period=7)
        valid = atr.dropna()
        assert (valid >= 0.0).all()

    def test_warmup_rows_are_nan(self):
        prices = pd.Series(np.arange(200, dtype=float))
        high   = prices + 0.5
        low    = prices - 0.5
        period = 14
        atr = _compute_atr_series(high, low, prices, period=period)
        warmup = period * 3 - 1
        assert atr.iloc[:warmup].isna().all()
        assert atr.iloc[warmup:].notna().all()

    def test_larger_price_range_produces_higher_atr(self):
        """ATR of a high-range series must exceed ATR of a low-range series."""
        n = 200
        prices = pd.Series(np.full(n, 500.0))

        low_range_atr = _compute_atr_series(
            prices + 0.1, prices - 0.1, prices, period=7
        ).dropna().mean()

        high_range_atr = _compute_atr_series(
            prices + 2.0, prices - 2.0, prices, period=7
        ).dropna().mean()

        assert high_range_atr > low_range_atr

    def test_flat_price_series_produces_finite_atr(self):
        """Flat price with constant high/low spread must not produce Inf or NaN after warmup."""
        prices = pd.Series(np.full(100, 500.0))
        atr = _compute_atr_series(prices + 0.5, prices - 0.5, prices, period=7)
        valid = atr.dropna()
        assert np.isfinite(valid.values).all()


# ── MFI ───────────────────────────────────────────────────────────────────────

class TestComputeMfi:
    def test_mfi_stays_within_bounds(self):
        rng   = np.random.default_rng(seed=42)
        prices = pd.Series(500 + rng.normal(0, 1, 300).cumsum())
        high   = prices + 0.5
        low    = prices - 0.5
        volume = pd.Series(np.full(300, 1_000))
        mfi = _compute_mfi_series(high, low, prices, volume, period=7)
        valid = mfi.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_warmup_rows_are_nan(self):
        n      = 200
        prices = pd.Series(np.arange(1, n + 1, dtype=float))
        high   = prices + 0.5
        low    = prices - 0.5
        volume = pd.Series(np.full(n, 1_000))
        period = 14
        mfi = _compute_mfi_series(high, low, prices, volume, period=period)
        # MFI uses rolling(period).sum() → first period-1 rows NaN.
        warmup = period - 1
        assert mfi.iloc[:warmup].isna().all()

    def test_zero_volume_gives_neutral_mfi(self):
        """All-zero volume means both money flows are 0 → MFI = 50 (neutral)."""
        n      = 100
        prices = pd.Series(np.arange(1, n + 1, dtype=float))
        high   = prices + 0.5
        low    = prices - 0.5
        volume = pd.Series(np.zeros(n))
        mfi = _compute_mfi_series(high, low, prices, volume, period=7)
        valid = mfi.dropna()
        assert np.allclose(valid.values, 50.0, atol=1e-4)

    def test_strictly_rising_price_gives_high_mfi(self):
        """All positive money flow (price only rises) → MFI near 100."""
        n      = 200
        prices = pd.Series(np.arange(1, n + 1, dtype=float))
        high   = prices + 0.1
        low    = prices - 0.1
        volume = pd.Series(np.full(n, 1_000))
        mfi = _compute_mfi_series(high, low, prices, volume, period=7)
        valid = mfi.dropna()
        # All typical prices are rising → all flow is positive → MFI = 100.
        assert np.allclose(valid.values, 100.0, atol=1e-4)


# ── Realized volatility ───────────────────────────────────────────────────────

class TestComputeRealizedVol:
    def test_realized_vol_is_non_negative_after_warmup(self):
        rng   = np.random.default_rng(seed=42)
        price = pd.Series(np.abs(500 + rng.normal(0, 1, 200).cumsum()))
        rvol  = _compute_realized_vol_series(price, period=7)
        valid = rvol.dropna()
        assert (valid >= 0.0).all()

    def test_warmup_rows_are_nan(self):
        price  = pd.Series(np.arange(1, 201, dtype=float))
        period = 14
        rvol   = _compute_realized_vol_series(price, period=period)
        # rolling(14).std() on log_returns (which have 1 leading NaN): first 14 rows NaN.
        assert rvol.iloc[:period].isna().all()

    def test_flat_price_has_zero_realized_vol(self):
        """Constant price → zero log returns → std = 0 (or NaN for window of 0s)."""
        price = pd.Series(np.full(100, 500.0))
        rvol  = _compute_realized_vol_series(price, period=7)
        valid = rvol.dropna()
        # std of a constant series is 0.
        assert np.allclose(valid.values, 0.0, atol=1e-10)

    def test_higher_volatility_produces_higher_realized_vol(self):
        """More volatile price series must produce higher average realized vol."""
        n = 200
        low_vol_price  = pd.Series(500 + np.random.default_rng(0).normal(0, 0.01, n).cumsum())
        high_vol_price = pd.Series(500 + np.random.default_rng(0).normal(0, 1.0,  n).cumsum())

        rvol_low  = _compute_realized_vol_series(low_vol_price,  period=14).dropna().mean()
        rvol_high = _compute_realized_vol_series(high_vol_price, period=14).dropna().mean()

        assert rvol_high > rvol_low


# ── compute_indicators (integration) ─────────────────────────────────────────

class TestComputeIndicators:
    def test_output_has_all_26_feature_columns(self):
        frame  = _make_aligned_frame(n_bars=200)
        result = compute_indicators(frame, config=DEFAULT_CONFIG)
        expected_columns = {
            "ema_5", "ema_9", "ema_12",
            "rsi_7", "rsi_14", "rsi_28",
            "vwap_deviation", "price_momentum_5", "volume_momentum_5",
            "bb_pct_b_7",  "bb_pct_b_14",  "bb_pct_b_28",
            "bb_width_7",  "bb_width_14",  "bb_width_28",
            "atr_7",       "atr_14",       "atr_28",
            "mfi_7",       "mfi_14",       "mfi_28",
            "realized_vol_7", "realized_vol_14", "realized_vol_28",
        }
        assert expected_columns.issubset(set(result.columns))

    def test_no_inf_values_in_output(self):
        frame  = _make_aligned_frame(n_bars=200)
        result = compute_indicators(frame, config=DEFAULT_CONFIG)
        numeric_cols = result.select_dtypes(include="number")
        assert not np.isinf(numeric_cols.values).any()

    def test_rsi_within_bounds_after_warmup(self):
        frame  = _make_aligned_frame(n_bars=300, price_trend=0.01)
        result = compute_indicators(frame, config=DEFAULT_CONFIG)
        for period in DEFAULT_CONFIG.indicator_periods:
            rsi_valid = result[f"rsi_{period}"].dropna()
            assert (rsi_valid >= 0.0).all(), f"rsi_{period} below 0"
            assert (rsi_valid <= 100.0).all(), f"rsi_{period} above 100"

    def test_warmup_rows_contain_nan(self):
        """The first 83 rows should have NaN in at least one indicator column
        (ATR-28 and RSI-28 have the longest warmup: 28×3-1 = 83 bars)."""
        frame  = _make_aligned_frame(n_bars=200)
        result = compute_indicators(frame, config=DEFAULT_CONFIG)
        # ATR-28 has warmup = 28*3-1 = 83 bars.
        assert result["atr_28"].iloc[:83].isna().all()

    def test_row_count_is_unchanged(self):
        frame  = _make_aligned_frame(n_bars=200)
        result = compute_indicators(frame, config=DEFAULT_CONFIG)
        assert len(result) == len(frame)

    def test_edge_case_flat_price_and_zero_volume(self):
        """Flat price with zero volume must not raise or produce Inf."""
        frame  = _make_aligned_frame(n_bars=200, price_trend=0.0, volume=0)
        result = compute_indicators(frame, config=DEFAULT_CONFIG)
        numeric_cols = result.select_dtypes(include="number")
        assert not np.isinf(numeric_cols.values).any()

    def test_mfi_within_bounds_after_warmup(self):
        frame  = _make_aligned_frame(n_bars=200, price_trend=0.01)
        result = compute_indicators(frame, config=DEFAULT_CONFIG)
        for period in DEFAULT_CONFIG.indicator_periods:
            mfi_valid = result[f"mfi_{period}"].dropna()
            assert (mfi_valid >= 0.0).all(), f"mfi_{period} below 0"
            assert (mfi_valid <= 100.0).all(), f"mfi_{period} above 100"

    def test_bb_pct_b_within_bounds_after_warmup(self):
        frame  = _make_aligned_frame(n_bars=200, price_trend=0.01)
        result = compute_indicators(frame, config=DEFAULT_CONFIG)
        for period in DEFAULT_CONFIG.indicator_periods:
            bb_valid = result[f"bb_pct_b_{period}"].dropna()
            assert (bb_valid >= 0.0).all(), f"bb_pct_b_{period} below 0"
            assert (bb_valid <= 1.0).all(), f"bb_pct_b_{period} above 1"

    def test_atr_is_non_negative_after_warmup(self):
        frame  = _make_aligned_frame(n_bars=200, price_trend=0.01)
        result = compute_indicators(frame, config=DEFAULT_CONFIG)
        for period in DEFAULT_CONFIG.indicator_periods:
            atr_valid = result[f"atr_{period}"].dropna()
            assert (atr_valid >= 0.0).all(), f"atr_{period} has negative values"
