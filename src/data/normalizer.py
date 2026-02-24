"""
Per-feature z-score normalization for 60-bar input windows.

Normalization is computed independently for each of the N_features columns
over the time axis of a single 60-bar window, so no feature dominates by
magnitude (e.g. raw price ~500 vs RSI 0-100).

The inverse transform recovers original-dollar-scale SPY close prices from
normalized model predictions.
"""

import numpy as np

# Prevents division by zero for features with near-zero variance
# (e.g. very flat price within a 60-bar window).
_NORMALIZATION_EPSILON = 1e-6

# Index of the spy_close column in the feature vector.
# Must match the ordering in sequence_builder.FEATURE_COLUMN_NAMES.
_SPY_CLOSE_FEATURE_INDEX = 0


def normalize_window(
    window: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize a single input window per feature column.

    Parameters
    ----------
    window : np.ndarray
        Shape (sequence_length, n_features). Must be finite (no NaN or Inf).

    Returns
    -------
    normalized_window : np.ndarray
        Shape (sequence_length, n_features) as float32.
        Each feature column has mean ≈ 0 and std ≈ 1 across the time axis.
    feature_means : np.ndarray
        Shape (n_features,) as float32. Per-feature means from this window.
    feature_stds : np.ndarray
        Shape (n_features,) as float32. Per-feature standard deviations.

    Raises
    ------
    ValueError
        If the input contains NaN or Inf values, or if the output does.
    """
    if not np.isfinite(window).all():
        raise ValueError(
            "Input window contains NaN or Inf values before normalization. "
            "Ensure indicator warmup rows are dropped before calling normalize_window()."
        )

    feature_means = window.mean(axis=0)
    feature_stds  = window.std(axis=0)

    normalized_window = (window - feature_means) / (feature_stds + _NORMALIZATION_EPSILON)

    if not np.isfinite(normalized_window).all():
        raise ValueError(
            "Normalized window contains NaN or Inf values. "
            "Check for constant-valued features in the input window."
        )

    return (
        normalized_window.astype(np.float32),
        feature_means.astype(np.float32),
        feature_stds.astype(np.float32),
    )


def denormalize_predictions(
    normalized_predictions: np.ndarray,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    spy_close_feature_index: int = _SPY_CLOSE_FEATURE_INDEX,
) -> np.ndarray:
    """
    Recover original-dollar-scale SPY close prices from normalized predictions.

    The predictions are assumed to have been produced by a model trained on
    windows normalized by normalize_window(). The inverse transform uses the
    spy_close column's mean and std from that same normalization.

    Parameters
    ----------
    normalized_predictions : np.ndarray
        Shape (prediction_steps,). Model output in normalized space.
    feature_means : np.ndarray
        Shape (n_features,). Returned by normalize_window() for the input window.
    feature_stds : np.ndarray
        Shape (n_features,). Returned by normalize_window() for the input window.
    spy_close_feature_index : int
        Column index of spy_close in the feature vector (default 0).

    Returns
    -------
    np.ndarray
        Shape (prediction_steps,) as float32. SPY close prices in dollar scale.
    """
    spy_close_mean = feature_means[spy_close_feature_index]
    spy_close_std  = feature_stds[spy_close_feature_index]
    return (normalized_predictions * (spy_close_std + _NORMALIZATION_EPSILON) + spy_close_mean).astype(np.float32)
