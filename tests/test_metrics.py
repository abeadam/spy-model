"""
Tests for evaluation metrics.
These run fast and must always pass before any result is trusted.
"""

import numpy as np
import pytest
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_per_step_metrics,
    directional_accuracy,
    mean_absolute_error,
    r_squared,
    root_mean_squared_error,
)


class TestMeanAbsoluteError:
    def test_perfect_prediction_returns_zero(self):
        values = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_error(values, values) == 0.0

    def test_known_values(self):
        actual    = np.array([10.0, 20.0, 30.0])
        predicted = np.array([12.0, 18.0, 33.0])
        # |−2| + |2| + |−3| = 7  →  avg = 7/3
        assert mean_absolute_error(actual, predicted) == pytest.approx(7 / 3, rel=1e-5)


class TestRootMeanSquaredError:
    def test_perfect_prediction_returns_zero(self):
        values = np.array([5.0, 10.0, 15.0])
        assert root_mean_squared_error(values, values) == 0.0

    def test_known_values(self):
        actual    = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 2.0, 2.0])
        # errors: 1, 0, 1  →  mse = 2/3  →  rmse = sqrt(2/3)
        expected = np.sqrt(2 / 3)
        assert root_mean_squared_error(actual, predicted) == pytest.approx(expected, rel=1e-5)


class TestRSquared:
    def test_perfect_prediction_returns_one(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert r_squared(values, values) == pytest.approx(1.0)

    def test_predicting_mean_returns_zero(self):
        actual    = np.array([1.0, 2.0, 3.0])
        mean_pred = np.full_like(actual, actual.mean())
        assert r_squared(actual, mean_pred) == pytest.approx(0.0, abs=1e-10)

    def test_constant_actual_returns_zero(self):
        actual    = np.array([5.0, 5.0, 5.0])
        predicted = np.array([4.0, 5.0, 6.0])
        assert r_squared(actual, predicted) == 0.0


class TestDirectionalAccuracy:
    def test_perfect_direction_returns_100(self):
        actual    = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.0, 2.1, 3.1, 4.1])
        assert directional_accuracy(actual, predicted) == pytest.approx(100.0)

    def test_opposite_direction_returns_zero(self):
        actual    = np.array([1.0, 2.0, 3.0])
        predicted = np.array([3.0, 2.0, 1.0])
        assert directional_accuracy(actual, predicted) == pytest.approx(0.0)

    def test_fifty_percent_accuracy(self):
        actual    = np.array([1.0, 2.0, 1.0, 2.0])
        predicted = np.array([1.0, 2.0, 2.0, 1.0])
        # diffs actual: +1, -1, +1  |  diffs pred: +1, 0, -1
        # correct: [True, False, False]  →  33.3%... let's use clearer values
        actual    = np.array([1.0, 3.0, 1.0])
        predicted = np.array([1.0, 3.0, 3.0])
        # diffs actual: +2, -2  |  diffs pred: +2, 0
        # sign: actual [+1, -1]  pred [+1, 0]
        # match: [True, False]  →  50%
        assert directional_accuracy(actual, predicted) == pytest.approx(50.0)


class TestComputeAllMetrics:
    def test_returns_all_expected_keys(self):
        values  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = compute_all_metrics(values, values)

        expected_keys = {"mae", "mape", "rmse", "r_squared", "directional_accuracy"}
        assert set(metrics.keys()) == expected_keys

    def test_perfect_predictions_give_best_scores(self):
        values  = np.array([100.0, 101.0, 99.0, 102.0, 98.0])
        metrics = compute_all_metrics(values, values)

        assert metrics["mae"]  == pytest.approx(0.0, abs=1e-10)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["r_squared"] == pytest.approx(1.0)
        assert metrics["directional_accuracy"] == pytest.approx(100.0)


class TestComputePerStepMetrics:
    """Tests for the multi-step metrics function required by FR-006."""

    NUMBER_OF_PREDICTION_STEPS = 5

    def _make_perfect_batch(self, num_samples: int = 10) -> np.ndarray:
        rng = np.random.default_rng(seed=0)
        return rng.uniform(low=400.0, high=500.0, size=(num_samples, self.NUMBER_OF_PREDICTION_STEPS)).astype(np.float32)

    def test_returns_all_per_step_and_aggregate_keys(self):
        values  = self._make_perfect_batch()
        metrics = compute_per_step_metrics(values, values)

        for step in range(1, self.NUMBER_OF_PREDICTION_STEPS + 1):
            assert f"mae_step_{step}"                  in metrics
            assert f"rmse_step_{step}"                 in metrics
            assert f"r2_step_{step}"                   in metrics
            assert f"directional_accuracy_step_{step}" in metrics

        assert "mae_aggregate"                  in metrics
        assert "rmse_aggregate"                 in metrics
        assert "r2_aggregate"                   in metrics
        assert "directional_accuracy_aggregate" in metrics

    def test_perfect_predictions_give_zero_error_per_step(self):
        values  = self._make_perfect_batch()
        metrics = compute_per_step_metrics(values, values)

        for step in range(1, self.NUMBER_OF_PREDICTION_STEPS + 1):
            assert metrics[f"mae_step_{step}"]  == pytest.approx(0.0, abs=1e-5)
            assert metrics[f"rmse_step_{step}"] == pytest.approx(0.0, abs=1e-5)

    def test_perfect_predictions_give_r2_of_one_per_step(self):
        actual = self._make_perfect_batch()
        # Add small noise so total_variance > 0 (r² is undefined for constant sequences)
        noisy  = actual + np.random.default_rng(1).normal(0, 0.01, actual.shape)
        metrics = compute_per_step_metrics(noisy, noisy)

        for step in range(1, self.NUMBER_OF_PREDICTION_STEPS + 1):
            assert metrics[f"r2_step_{step}"] == pytest.approx(1.0, abs=1e-5)

    def test_aggregate_mae_consistent_with_flat_computation(self):
        rng     = np.random.default_rng(seed=2)
        actual    = rng.uniform(400.0, 500.0, (20, self.NUMBER_OF_PREDICTION_STEPS))
        predicted = actual + rng.normal(0, 1.0, actual.shape)

        metrics       = compute_per_step_metrics(actual, predicted)
        expected_mae  = float(np.mean(np.abs(actual.flatten() - predicted.flatten())))

        assert metrics["mae_aggregate"] == pytest.approx(expected_mae, rel=1e-5)

    def test_raises_on_1d_input(self):
        values_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="2-D"):
            compute_per_step_metrics(values_1d, values_1d)

    def test_raises_on_shape_mismatch(self):
        actual    = np.ones((10, 5))
        predicted = np.ones((10, 4))
        with pytest.raises(ValueError, match="same shape"):
            compute_per_step_metrics(actual, predicted)

    def test_total_key_count_is_correct(self):
        values  = self._make_perfect_batch()
        metrics = compute_per_step_metrics(values, values)

        # 4 metrics × 5 steps + 4 aggregate = 24 keys
        expected_key_count = 4 * self.NUMBER_OF_PREDICTION_STEPS + 4
        assert len(metrics) == expected_key_count
