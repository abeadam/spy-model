"""
Tests for evaluation metrics.
These run fast and must always pass before any result is trusted.
"""

import numpy as np
import pytest
from src.evaluation.metrics import (
    compute_all_metrics,
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
