"""
Measure and log inference latency for every SpyPredictor.predict() call.

Every call is timed with time.perf_counter. Calls that exceed
config.max_inference_latency_ms are flagged with a logged warning so
violations are visible in the experiment log without requiring a code change.
"""

import time
from typing import Callable

import numpy as np

from src.utils.config import DEFAULT_CONFIG, TrainingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def measure_inference_latency(
    predict_fn: Callable[[np.ndarray], tuple[np.ndarray, str, float]],
    raw_input_window: np.ndarray,
    config: TrainingConfig = DEFAULT_CONFIG,
) -> tuple[tuple[np.ndarray, str, float], float, bool]:
    """
    Wrap a single predict() call with wall-clock timing and violation detection.

    Parameters
    ----------
    predict_fn : Callable
        The model's predict() method (or any callable with the same signature).
    raw_input_window : np.ndarray
        Shape (sequence_length, n_features). Passed directly to predict_fn.
    config : TrainingConfig
        Supplies max_inference_latency_ms for violation threshold.

    Returns
    -------
    prediction : tuple[np.ndarray, str, float]
        The full return value of predict_fn (predicted_prices, directional_signal,
        inner_latency_ms). The inner_latency_ms is the timing from inside
        predict() itself; outer_latency_ms below is measured by this wrapper.
    outer_latency_ms : float
        Wall-clock time from just before the predict_fn call to just after,
        in milliseconds. This is the value logged and checked for violations.
    is_violation : bool
        True if outer_latency_ms exceeds config.max_inference_latency_ms.
    """
    call_start = time.perf_counter()
    prediction = predict_fn(raw_input_window)
    call_end   = time.perf_counter()

    outer_latency_ms = (call_end - call_start) * 1_000.0
    is_violation     = outer_latency_ms > config.max_inference_latency_ms

    if is_violation:
        logger.warning(
            "Inference latency violation: %.2f ms > %.1f ms threshold.",
            outer_latency_ms,
            config.max_inference_latency_ms,
        )

    return prediction, outer_latency_ms, is_violation
