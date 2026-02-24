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


def compute_per_step_metrics(
    actual_values: np.ndarray,
    predicted_values: np.ndarray,
) -> dict[str, float]:
    """
    Compute MAE, RMSE, R², and directional accuracy for each prediction step
    individually, plus aggregate metrics across all steps.

    Parameters
    ----------
    actual_values : np.ndarray
        Shape (N, prediction_steps) — ground-truth SPY close prices.
    predicted_values : np.ndarray
        Shape (N, prediction_steps) — model-predicted SPY close prices.

    Returns
    -------
    dict[str, float]
        Keys follow the pattern:
          mae_step_{k}, rmse_step_{k}, r2_step_{k}, directional_accuracy_step_{k}
          for k in 1..prediction_steps, plus:
          mae_aggregate, rmse_aggregate, r2_aggregate, directional_accuracy_aggregate
    """
    if actual_values.ndim != 2 or predicted_values.ndim != 2:
        raise ValueError(
            f"Both arrays must be 2-D (N, prediction_steps). "
            f"Got actual={actual_values.shape}, predicted={predicted_values.shape}."
        )
    if actual_values.shape != predicted_values.shape:
        raise ValueError(
            f"actual_values and predicted_values must have the same shape. "
            f"Got {actual_values.shape} vs {predicted_values.shape}."
        )

    number_of_steps = actual_values.shape[1]
    result: dict[str, float] = {}

    for step_index in range(number_of_steps):
        step_label = step_index + 1
        actual_step    = actual_values[:, step_index]
        predicted_step = predicted_values[:, step_index]

        result[f"mae_step_{step_label}"]                  = mean_absolute_error(actual_step, predicted_step)
        result[f"rmse_step_{step_label}"]                 = root_mean_squared_error(actual_step, predicted_step)
        result[f"r2_step_{step_label}"]                   = r_squared(actual_step, predicted_step)
        result[f"directional_accuracy_step_{step_label}"] = directional_accuracy(actual_step, predicted_step)

    # Aggregate metrics treat all steps as a single flat sequence.
    actual_flat    = actual_values.flatten()
    predicted_flat = predicted_values.flatten()

    result["mae_aggregate"]                  = mean_absolute_error(actual_flat, predicted_flat)
    result["rmse_aggregate"]                 = root_mean_squared_error(actual_flat, predicted_flat)
    result["r2_aggregate"]                   = r_squared(actual_flat, predicted_flat)
    result["directional_accuracy_aggregate"] = directional_accuracy(actual_flat, predicted_flat)

    return result
