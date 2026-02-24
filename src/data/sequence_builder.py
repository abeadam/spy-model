"""
Build stride-1 sliding input/target sequence pairs from a day's feature DataFrame.

Each (input_sequence, target) pair consists of:
    input_sequence : (120, 26) float32 — normalized 120-bar feature window
    target         : (5,)      float32 — next 5 SPY close returns from bar 120

Target formula: target[k] = (close[t+k] - close[t]) / close[t]  for k=1..5
where close[t] is the last bar of the input window (bar 120). Percent changes are
scale-invariant across price levels and do not require denormalization at inference.

Windows must not span trading-day boundaries; each call to
build_sequences_for_day() covers exactly one day's data.

Days with insufficient bars after indicator warmup are returned with zero
sequences (empty arrays) so the caller can skip them silently.
"""

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.data.normalizer import normalize_window
from src.utils.config import DEFAULT_CONFIG, TrainingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Column names in the exact order they form the 26-feature input vector.
# This ordering is the contract between sequence_builder, model, and normalizer.
FEATURE_COLUMN_NAMES: list[str] = [
    "spy_close",           # index 0  — also used to extract raw targets
    "vix_close",           # index 1
    "ema_5",               # index 2
    "ema_9",               # index 3
    "ema_12",              # index 4
    "rsi_7",               # index 5
    "rsi_14",              # index 6
    "rsi_28",              # index 7
    "vwap_deviation",      # index 8
    "price_momentum_5",    # index 9
    "volume_momentum_5",   # index 10
    "bb_pct_b_7",          # index 11
    "bb_pct_b_14",         # index 12
    "bb_pct_b_28",         # index 13
    "bb_width_7",          # index 14
    "bb_width_14",         # index 15
    "bb_width_28",         # index 16
    "atr_7",               # index 17
    "atr_14",              # index 18
    "atr_28",              # index 19
    "mfi_7",               # index 20
    "mfi_14",              # index 21
    "mfi_28",              # index 22
    "realized_vol_7",      # index 23
    "realized_vol_14",     # index 24
    "realized_vol_28",     # index 25
]

_SPY_CLOSE_COLUMN_INDEX = FEATURE_COLUMN_NAMES.index("spy_close")

_VALID_TRANSFORMS = frozenset({
    "percent_change",
    "basis_points",
    "signed_log",
    "vol_normalized",
    "vol_normalized_log",
    "vol_normalized_log_bp",      # log return in bp ÷ local vol in raw units → bp-Sharpe
    "vol_normalized_log_prevday", # log return ÷ prior full-day realized vol (stable denominator)
})


def _compute_day_realized_vol(spy_close_values: np.ndarray) -> float:
    """
    Compute realized volatility for a full trading day as std of log returns.

    Uses all bars in the day (before indicator warmup filtering) for a stable
    vol estimate unaffected by the post-warmup bar count.

    Parameters
    ----------
    spy_close_values : np.ndarray
        All SPY close prices for the day (float64 or float32).

    Returns
    -------
    float
        Std of log returns, plus a small epsilon to avoid division by zero.
    """
    log_returns = np.diff(np.log(spy_close_values.astype(np.float64)))
    return float(np.std(log_returns)) + 1e-8


def _apply_target_transform(
    future_closes: np.ndarray,
    anchor_close: float,
    window_closes: np.ndarray,
    transform: str,
    prev_day_vol: Optional[float] = None,
) -> np.ndarray:
    """
    Convert future absolute close prices into the configured target representation.

    Parameters
    ----------
    future_closes : np.ndarray
        Shape (prediction_steps,). Absolute SPY close prices for the target bars.
    anchor_close : float
        SPY close of the last bar of the input window. Used as reference.
    window_closes : np.ndarray
        Shape (sequence_length,). SPY close prices in the input window. Used to
        compute local realized volatility for the vol-normalized transforms.
    transform : str
        One of the transforms listed in _VALID_TRANSFORMS.
    prev_day_vol : float, optional
        Prior full-day realized vol. Required for vol_normalized_log_prevday.

    Returns
    -------
    np.ndarray
        Shape (prediction_steps,) float32 targets in the requested representation.
    """
    if transform not in _VALID_TRANSFORMS:
        raise ValueError(
            f"Unknown target_transform {transform!r}. "
            f"Valid options: {sorted(_VALID_TRANSFORMS)}"
        )

    pct_changes = (future_closes - anchor_close) / anchor_close

    if transform == "percent_change":
        return pct_changes.astype(np.float32)

    if transform == "basis_points":
        return (pct_changes * 10_000.0).astype(np.float32)

    if transform == "signed_log":
        return (np.sign(pct_changes) * np.log1p(np.abs(pct_changes))).astype(np.float32)

    # Vol-normalized variants: need local realized vol from the input window.
    if transform in ("vol_normalized", "vol_normalized_log", "vol_normalized_log_bp"):
        if transform == "vol_normalized":
            window_returns = np.diff(window_closes) / window_closes[:-1]
            local_vol      = float(np.std(window_returns)) + 1e-8
            return (pct_changes / local_vol).astype(np.float32)

        # Both log variants share the same local vol (std of window log returns).
        window_log_returns = np.diff(np.log(window_closes.astype(np.float64)))
        local_vol          = float(np.std(window_log_returns)) + 1e-8
        log_changes        = np.log(future_closes / anchor_close)

        if transform == "vol_normalized_log":
            return (log_changes / local_vol).astype(np.float32)

        # vol_normalized_log_bp: numerator in basis points.
        return (10_000.0 * log_changes / local_vol).astype(np.float32)

    if transform == "vol_normalized_log_prevday":
        if prev_day_vol is None:
            raise ValueError(
                "prev_day_vol is required for vol_normalized_log_prevday transform."
            )
        log_changes = np.log(future_closes / anchor_close)
        return (log_changes / (prev_day_vol + 1e-8)).astype(np.float32)

    # Unreachable — _VALID_TRANSFORMS guard above covers all cases.
    raise RuntimeError(f"Unhandled transform: {transform!r}")


def build_sequences_for_day(
    daily_feature_frame: pd.DataFrame,
    trading_date: date,
    config: TrainingConfig = DEFAULT_CONFIG,
    prev_day_realized_vol: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build all valid (input_sequence, target) pairs for one trading day.

    Drops rows with any NaN value (indicator warmup period), then applies a
    stride-1 sliding window over the remaining bars. Sequences never cross
    day boundaries because this function operates on a single day's frame.

    Parameters
    ----------
    daily_feature_frame : pd.DataFrame
        Output of indicator_computer.compute_indicators() for one trading day.
        Must contain all columns listed in FEATURE_COLUMN_NAMES.
    trading_date : date
        Used only for warning messages — no computation depends on it.
    config : TrainingConfig
        Supplies sequence_length, prediction_steps, n_features, target_transform.
    prev_day_realized_vol : float, optional
        Prior full-day realized vol required when target_transform is
        'vol_normalized_log_prevday'. If None and that transform is active,
        returns empty arrays (first day of data — skipped silently).

    Returns
    -------
    sequences : np.ndarray
        Shape (N, sequence_length, n_features) float32. Normalized windows.
    targets : np.ndarray
        Shape (N, prediction_steps) float32. Targets in the configured encoding.

    Returns empty arrays (shape (0, ...)) if the day has insufficient bars or
    if prev_day_realized_vol is None for the prevday transform.
    """
    sequence_length  = config.sequence_length
    prediction_steps = config.prediction_steps
    n_features       = config.n_features
    minimum_bars     = sequence_length + prediction_steps

    # For the prevday transform, skip the first trading day (no prior vol available).
    if config.target_transform == "vol_normalized_log_prevday" and prev_day_realized_vol is None:
        logger.debug(
            "%s: no prev_day_realized_vol for vol_normalized_log_prevday — skipping.",
            trading_date,
        )
        return (
            np.empty((0, sequence_length, n_features), dtype=np.float32),
            np.empty((0, prediction_steps),            dtype=np.float32),
        )

    # Drop warmup rows — any column containing NaN is invalid.
    clean_frame = daily_feature_frame[FEATURE_COLUMN_NAMES].dropna()

    if len(clean_frame) < minimum_bars:
        logger.warning(
            "%s: %d valid bars after warmup (need ≥ %d) — no sequences produced.",
            trading_date,
            len(clean_frame),
            minimum_bars,
        )
        return (
            np.empty((0, sequence_length, n_features), dtype=np.float32),
            np.empty((0, prediction_steps),            dtype=np.float32),
        )

    feature_matrix    = clean_frame.to_numpy(dtype=np.float32)
    number_of_windows = len(feature_matrix) - minimum_bars + 1

    sequences = np.empty((number_of_windows, sequence_length, n_features), dtype=np.float32)
    targets   = np.empty((number_of_windows, prediction_steps),            dtype=np.float32)

    for window_index in range(number_of_windows):
        window_start = window_index
        window_end   = window_index + sequence_length
        target_end   = window_end   + prediction_steps

        raw_window              = feature_matrix[window_start:window_end]
        normalized_window, _, _ = normalize_window(raw_window)

        sequences[window_index] = normalized_window
        anchor_close            = feature_matrix[window_end - 1, _SPY_CLOSE_COLUMN_INDEX]
        future_closes           = feature_matrix[window_end:target_end, _SPY_CLOSE_COLUMN_INDEX]
        window_closes           = feature_matrix[window_start:window_end, _SPY_CLOSE_COLUMN_INDEX]
        targets[window_index]   = _apply_target_transform(
            future_closes, anchor_close, window_closes,
            config.target_transform, prev_day_vol=prev_day_realized_vol,
        )

    return sequences, targets


def build_all_sequences(
    daily_feature_frames: dict[date, pd.DataFrame],
    config: TrainingConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray, list[date]]:
    """
    Build all sequence pairs across all trading days.

    For the 'vol_normalized_log_prevday' transform, each day's sequences use
    the prior full-day's realized vol as the normalizing denominator. Day vol
    is computed from all spy_close bars in the raw indicator frame (before
    warmup filtering) for a stable estimate. The first trading day always
    produces zero sequences because no prior vol is available.

    Parameters
    ----------
    daily_feature_frames : dict[date, pd.DataFrame]
        Mapping from trading date to indicator-enriched bar DataFrame.
        Produced by indicator_computer.compute_indicators() for each day.
    config : TrainingConfig

    Returns
    -------
    all_sequences : np.ndarray
        Shape (total_N, sequence_length, n_features).
    all_targets : np.ndarray
        Shape (total_N, prediction_steps).
    sequence_dates : list[date]
        The trading date each sequence belongs to (len == total_N).
        Used downstream for chronological train/val/test splitting.

    Raises
    ------
    RuntimeError
        If no sequences were produced across all days.
    """
    sorted_dates = sorted(daily_feature_frames)

    # Pre-compute per-day realized vol when the prevday transform is active.
    # Uses all bars in the raw frame (before dropna) for a stable denominator.
    daily_realized_vol: dict[date, float] = {}
    if config.target_transform == "vol_normalized_log_prevday":
        for trading_date in sorted_dates:
            spy_close_values = daily_feature_frames[trading_date]["spy_close"].values
            daily_realized_vol[trading_date] = _compute_day_realized_vol(spy_close_values)

    all_sequences_list: list[np.ndarray] = []
    all_targets_list:   list[np.ndarray] = []
    sequence_dates:     list[date]       = []

    for i, trading_date in enumerate(sorted_dates):
        prev_day_realized_vol: Optional[float] = None
        if config.target_transform == "vol_normalized_log_prevday" and i > 0:
            prev_trading_date     = sorted_dates[i - 1]
            prev_day_realized_vol = daily_realized_vol.get(prev_trading_date)

        day_sequences, day_targets = build_sequences_for_day(
            daily_feature_frames[trading_date],
            trading_date,
            config,
            prev_day_realized_vol=prev_day_realized_vol,
        )
        if day_sequences.shape[0] == 0:
            continue

        all_sequences_list.append(day_sequences)
        all_targets_list.append(day_targets)
        sequence_dates.extend([trading_date] * day_sequences.shape[0])

    if not all_sequences_list:
        raise RuntimeError(
            "No sequences were produced across any trading day. "
            "Check that raw data is accessible and meets minimum bar requirements."
        )

    all_sequences = np.concatenate(all_sequences_list, axis=0)
    all_targets   = np.concatenate(all_targets_list,   axis=0)

    logger.info(
        "Built %d total sequences from %d trading days.",
        len(all_sequences),
        len(daily_feature_frames),
    )
    return all_sequences, all_targets, sequence_dates
