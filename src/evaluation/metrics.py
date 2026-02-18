"""
Evaluation metrics for model performance.
All metric computations live here so evaluation logic is never duplicated.
"""

import numpy as np


def mean_absolute_error(actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
    return float(np.mean(np.abs(actual_values - predicted_values)))


def mean_absolute_percentage_error(
    actual_values: np.ndarray,
    predicted_values: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """MAPE, protected against division by zero with epsilon."""
    return float(np.mean(np.abs((actual_values - predicted_values) / (actual_values + epsilon))) * 100)


def root_mean_squared_error(actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual_values - predicted_values) ** 2)))


def r_squared(actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """
    Coefficient of determination (R²).
    1.0 is perfect prediction. Negative means worse than predicting the mean.
    """
    total_variance = np.sum((actual_values - np.mean(actual_values)) ** 2)
    residual_variance = np.sum((actual_values - predicted_values) ** 2)
    if total_variance == 0:
        return 0.0
    return float(1 - (residual_variance / total_variance))


def directional_accuracy(actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """
    Percentage of predictions where the model got the direction of movement correct.
    Computed on consecutive-day differences.
    """
    actual_direction    = np.sign(np.diff(actual_values))
    predicted_direction = np.sign(np.diff(predicted_values))
    return float(np.mean(actual_direction == predicted_direction) * 100)


def compute_all_metrics(
    actual_values: np.ndarray,
    predicted_values: np.ndarray,
) -> dict[str, float]:
    """Compute the full suite of evaluation metrics in one call."""
    return {
        "mae":                  mean_absolute_error(actual_values, predicted_values),
        "mape":                 mean_absolute_percentage_error(actual_values, predicted_values),
        "rmse":                 root_mean_squared_error(actual_values, predicted_values),
        "r_squared":            r_squared(actual_values, predicted_values),
        "directional_accuracy": directional_accuracy(actual_values, predicted_values),
    }
