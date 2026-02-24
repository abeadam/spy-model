"""
Compute technical indicators for a single day's aligned bar DataFrame.

All computations use NumPy / pandas — no external TA libraries.

Indicators added to the DataFrame (one value per bar):
    ema_5              — 5-bar EMA of spy_close  (warmup: 15 bars)
    ema_9              — 9-bar EMA of spy_close  (warmup: 27 bars)
    ema_12             — 12-bar EMA of spy_close (warmup: 36 bars)
    rsi_7              — 7-period RSI, 0-100 scale  (warmup: 21 bars)
    rsi_14             — 14-period RSI, 0-100 scale (warmup: 41 bars)
    rsi_28             — 28-period RSI, 0-100 scale (warmup: 83 bars)
    vwap_deviation     — (spy_close - rolling_VWAP) / rolling_VWAP × 100
    price_momentum_5   — (spy_close[t] - spy_close[t-5]) / spy_close[t-5] × 100
    volume_momentum_5  — (spy_volume[t] - spy_volume[t-5]) / spy_volume[t-5] × 100
    bb_pct_b_{p}       — Bollinger %B for p in (7, 14, 28) (warmup: p-1 bars)
    bb_width_{p}       — Bollinger normalized bandwidth for p in (7, 14, 28)
    atr_{p}            — Average True Range for p in (7, 14, 28) (warmup: p×3-1 bars)
    mfi_{p}            — Money Flow Index for p in (7, 14, 28) (warmup: p-1 bars)
    realized_vol_{p}   — Rolling std of log returns for p in (7, 14, 28) (warmup: p bars)

Maximum warmup = 83 bars (ATR-28 and RSI-28). Warmup-period rows contain NaN
and must be dropped by the caller before building input sequences.
"""

import numpy as np
import pandas as pd

from src.utils.config import DEFAULT_CONFIG, TrainingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# EMA values are considered reliable after this many multiples of the period.
# Beyond 3× the period the initialization bias is < 5%.
_EMA_WARMUP_MULTIPLIER = 3

# RSI Wilder smoothing needs the same burn-in factor.
_RSI_WARMUP_MULTIPLIER = 3

# ATR uses Wilder's EMA — same warmup multiplier.
_ATR_WARMUP_MULTIPLIER = 3


def _compute_ema_series(price_series: pd.Series, period: int) -> pd.Series:
    """
    Exponential moving average with alpha = 2 / (period + 1).

    The first (period × EMA_WARMUP_MULTIPLIER − 1) values are set to NaN
    to flag the burn-in period where the seed value dominates.
    """
    warmup_bars = period * _EMA_WARMUP_MULTIPLIER
    ema_series  = price_series.ewm(span=period, adjust=False).mean()
    ema_series.iloc[: warmup_bars - 1] = np.nan
    return ema_series


def _compute_rsi_series(price_series: pd.Series, period: int) -> pd.Series:
    """
    RSI using Wilder smoothing (alpha = 1 / period).

    Returns values in the 0-100 range. The first
    (period × RSI_WARMUP_MULTIPLIER − 1) rows are set to NaN.

    Special cases:
        average_loss == 0 and average_gain  > 0  →  RSI = 100 (full overbought)
        average_gain == 0 and average_loss  > 0  →  RSI = 0   (full oversold)
        both == 0 (flat price)                   →  RSI = 50  (neutral)
    """
    price_delta  = price_series.diff()
    gains        = price_delta.clip(lower=0.0)
    losses       = (-price_delta).clip(lower=0.0)

    wilder_alpha = 1.0 / period
    average_gain = gains.ewm(alpha=wilder_alpha, adjust=False).mean()
    average_loss = losses.ewm(alpha=wilder_alpha, adjust=False).mean()

    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi_series = 100.0 - (100.0 / (1.0 + relative_strength))

    # When average_loss is 0 and gain > 0, RSI should be 100.
    rsi_series = rsi_series.where(average_loss != 0.0, other=100.0)
    # When both are 0 (flat price), RSI is neutral 50.
    rsi_series = rsi_series.where(
        ~((average_gain == 0.0) & (average_loss == 0.0)), other=50.0
    )

    warmup_bars = period * _RSI_WARMUP_MULTIPLIER
    rsi_series.iloc[: warmup_bars - 1] = np.nan
    return rsi_series


def _compute_vwap_deviation_series(
    spy_close: pd.Series,
    spy_high: pd.Series,
    spy_low: pd.Series,
    spy_volume: pd.Series,
    vwap_window: int,
) -> pd.Series:
    """
    Rolling VWAP deviation: (close − VWAP) / VWAP × 100.

    VWAP uses typical price = (high + low + close) / 3 weighted by spy_volume
    over a rolling `vwap_window`-bar window (min_periods=1).

    Bars where the rolling volume sum is zero fall back to using close as the
    VWAP, yielding a deviation of 0.
    """
    typical_price        = (spy_high + spy_low + spy_close) / 3.0
    rolling_price_volume = (typical_price * spy_volume).rolling(
        vwap_window, min_periods=1
    ).sum()
    rolling_volume = spy_volume.rolling(vwap_window, min_periods=1).sum()

    rolling_vwap = rolling_price_volume / rolling_volume.replace(0.0, np.nan)
    rolling_vwap = rolling_vwap.fillna(spy_close)  # fallback: deviation = 0

    return (spy_close - rolling_vwap) / rolling_vwap * 100.0


def _compute_momentum_series(values: pd.Series, period: int) -> pd.Series:
    """
    Percent-change momentum: (value[t] − value[t−period]) / value[t−period] × 100.

    The first `period` rows are naturally NaN from the lag.
    Zero denominators (flat values) yield NaN; callers must handle these.
    """
    lagged = values.shift(period)
    return (values - lagged) / lagged.replace(0.0, np.nan) * 100.0


def _compute_bollinger_bands_series(
    price_series: pd.Series,
    period: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Bollinger Bands %B and normalized bandwidth over `period` bars.

    %B = (close − lower_band) / (upper_band − lower_band), clipped to [0, 1].
    bandwidth = (upper_band − lower_band) / mid_band.

    Both use rolling(period) with default min_periods=period, so the first
    period-1 rows are NaN.

    Returns
    -------
    bb_pct_b : pd.Series — percent-B position within the band [0, 1].
    bb_width  : pd.Series — normalized band width (≥ 0).
    """
    rolling_mean = price_series.rolling(period).mean()
    rolling_std  = price_series.rolling(period).std()
    upper_band   = rolling_mean + 2.0 * rolling_std
    lower_band   = rolling_mean - 2.0 * rolling_std
    band_range   = upper_band - lower_band

    bb_pct_b = (price_series - lower_band) / band_range.replace(0.0, np.nan)
    bb_pct_b = bb_pct_b.clip(0.0, 1.0)

    bb_width = band_range / rolling_mean.replace(0.0, np.nan)

    return bb_pct_b, bb_width


def _compute_atr_series(
    spy_high: pd.Series,
    spy_low: pd.Series,
    spy_close: pd.Series,
    period: int,
) -> pd.Series:
    """
    Average True Range using Wilder's EMA (alpha = 1 / period).

    True Range = max(high − low, |high − prev_close|, |low − prev_close|).
    The first (period × ATR_WARMUP_MULTIPLIER − 1) rows are set to NaN.
    """
    prev_close = spy_close.shift(1)
    true_range = pd.concat(
        [
            spy_high - spy_low,
            (spy_high - prev_close).abs(),
            (spy_low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    wilder_alpha = 1.0 / period
    atr = true_range.ewm(alpha=wilder_alpha, adjust=False).mean()

    warmup_bars = period * _ATR_WARMUP_MULTIPLIER
    atr.iloc[: warmup_bars - 1] = np.nan
    return atr


def _compute_mfi_series(
    spy_high: pd.Series,
    spy_low: pd.Series,
    spy_close: pd.Series,
    spy_volume: pd.Series,
    period: int,
) -> pd.Series:
    """
    Money Flow Index over `period` bars.

    MFI = 100 × positive_money_flow / (positive_money_flow + negative_money_flow).
    The first period-1 rows are NaN (from rolling sum with min_periods=period).

    Special cases:
        negative flow sum == 0 and positive > 0  →  MFI = 100
        both == 0 (no volume)                    →  MFI = 50 (neutral)
    """
    typical_price   = (spy_high + spy_low + spy_close) / 3.0
    raw_money_flow  = typical_price * spy_volume
    price_direction = typical_price.diff().fillna(0.0)

    positive_flow = raw_money_flow.where(price_direction >= 0.0, other=0.0)
    negative_flow = raw_money_flow.where(price_direction < 0.0,  other=0.0)

    positive_flow_sum = positive_flow.rolling(period).sum()
    negative_flow_sum = negative_flow.rolling(period).sum()

    money_flow_ratio = positive_flow_sum / negative_flow_sum.replace(0.0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + money_flow_ratio))

    # When negative flow is 0 and positive > 0, MFI is 100.
    mfi = mfi.where(negative_flow_sum != 0.0, other=100.0)
    # When both are 0 (no volume), MFI is neutral 50.
    mfi = mfi.where(
        ~((positive_flow_sum == 0.0) & (negative_flow_sum == 0.0)), other=50.0
    )

    return mfi


def _compute_realized_vol_series(price_series: pd.Series, period: int) -> pd.Series:
    """
    Rolling standard deviation of log returns over `period` bars.

    The first `period` rows are NaN: one from the log-return diff, then
    rolling(period) needs `period` observations.
    """
    log_returns = np.log(price_series / price_series.shift(1))
    return log_returns.rolling(period).std()


def compute_indicators(
    aligned_daily_frame: pd.DataFrame,
    config: TrainingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Add all 26 technical indicator columns to an aligned daily bar DataFrame.

    The input must have columns: spy_open, spy_high, spy_low, spy_close,
    spy_volume, vix_close. The output has those same columns plus 20 indicator
    columns:
        ema_5, ema_9, ema_12
        rsi_7, rsi_14, rsi_28
        vwap_deviation, price_momentum_5, volume_momentum_5
        bb_pct_b_7,  bb_pct_b_14,  bb_pct_b_28
        bb_width_7,  bb_width_14,  bb_width_28
        atr_7,  atr_14,  atr_28
        mfi_7,  mfi_14,  mfi_28
        realized_vol_7,  realized_vol_14,  realized_vol_28

    Warmup-period rows contain NaN values and must be dropped before the
    DataFrame is used to build input sequences. Maximum warmup = 83 bars
    (ATR-28 and RSI-28).

    Parameters
    ----------
    aligned_daily_frame : pd.DataFrame
        Single-day aligned bars from bar_loader.load_aligned_daily_bars().
    config : TrainingConfig
        Supplies indicator periods, momentum_period, sequence_length (VWAP window).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with 20 indicator columns appended.
    """
    spy_close  = aligned_daily_frame["spy_close"].astype("float64")
    spy_high   = aligned_daily_frame["spy_high"].astype("float64")
    spy_low    = aligned_daily_frame["spy_low"].astype("float64")
    spy_volume = aligned_daily_frame["spy_volume"].astype("float64")

    result = aligned_daily_frame.copy()

    # ── EMA ──────────────────────────────────────────────────────────────────
    result["ema_5"]  = _compute_ema_series(spy_close, config.ema_short_period).astype("float32")
    result["ema_9"]  = _compute_ema_series(spy_close, config.ema_mid_period).astype("float32")
    result["ema_12"] = _compute_ema_series(spy_close, config.ema_long_period).astype("float32")

    # ── RSI for all indicator periods ─────────────────────────────────────────
    for period in config.indicator_periods:
        result[f"rsi_{period}"] = _compute_rsi_series(spy_close, period).astype("float32")

    # ── VWAP deviation ────────────────────────────────────────────────────────
    result["vwap_deviation"] = _compute_vwap_deviation_series(
        spy_close, spy_high, spy_low, spy_volume,
        vwap_window=config.sequence_length,
    ).astype("float32")

    # ── Momentum ──────────────────────────────────────────────────────────────
    result["price_momentum_5"] = _compute_momentum_series(
        spy_close, config.momentum_period
    ).astype("float32")

    # Zero-volume bars have undefined volume momentum; treat as no change (0).
    volume_momentum = _compute_momentum_series(spy_volume, config.momentum_period)
    result["volume_momentum_5"] = volume_momentum.fillna(0.0).astype("float32")

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    for period in config.indicator_periods:
        bb_pct_b, bb_width = _compute_bollinger_bands_series(spy_close, period)
        result[f"bb_pct_b_{period}"] = bb_pct_b.astype("float32")
        result[f"bb_width_{period}"] = bb_width.astype("float32")

    # ── ATR ───────────────────────────────────────────────────────────────────
    for period in config.indicator_periods:
        result[f"atr_{period}"] = _compute_atr_series(
            spy_high, spy_low, spy_close, period
        ).astype("float32")

    # ── MFI ───────────────────────────────────────────────────────────────────
    for period in config.indicator_periods:
        result[f"mfi_{period}"] = _compute_mfi_series(
            spy_high, spy_low, spy_close, spy_volume, period
        ).astype("float32")

    # ── Realized volatility ───────────────────────────────────────────────────
    for period in config.indicator_periods:
        result[f"realized_vol_{period}"] = _compute_realized_vol_series(
            spy_close, period
        ).astype("float32")

    return result
