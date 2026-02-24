"""
Tests for scripts/evaluate.py

Uses a tiny model and tiny synthetic dataset to keep tests fast (< 5s each).
All tests use CPU so MPS availability doesn't affect results.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.evaluate import compute_test_metrics, run_evaluation, sample_latency_stats
from src.data.dataset import SpyDataset
from src.model.checkpoint import save_checkpoint
from src.model.spy_predictor import SpyPredictor
from src.utils.config import TrainingConfig


# ── Shared constants and fixtures ─────────────────────────────────────────────

DEFAULT_N_FEATURES  = 26
DEFAULT_SEQ_LEN     = 120
DEFAULT_PRED_STEPS  = 5


def _make_tiny_config() -> TrainingConfig:
    return TrainingConfig(
        d_model=16,
        n_heads=2,
        n_encoder_layers=1,
        ffn_dim=32,
        dropout_rate=0.0,
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=1,
        early_stopping_patience=1,
        lr_scheduler_t0=10,
        random_seed=42,
        n_features=DEFAULT_N_FEATURES,
        sequence_length=DEFAULT_SEQ_LEN,
        prediction_steps=DEFAULT_PRED_STEPS,
    )


def _make_tiny_model(config: TrainingConfig) -> SpyPredictor:
    torch.manual_seed(0)
    return SpyPredictor(config=config, device=torch.device("cpu"))


def _make_synthetic_dataset(n_samples: int, seed: int = 0) -> SpyDataset:
    rng       = np.random.default_rng(seed=seed)
    sequences = rng.standard_normal((n_samples, DEFAULT_SEQ_LEN, DEFAULT_N_FEATURES)).astype(np.float32)
    targets   = rng.uniform(490.0, 510.0, (n_samples, DEFAULT_PRED_STEPS)).astype(np.float32)
    return SpyDataset(sequences, targets)


def _save_tiny_checkpoint(model: SpyPredictor, config: TrainingConfig, path: Path) -> None:
    """Save a minimal checkpoint so run_evaluation can load it."""
    save_checkpoint(
        model=model,
        config=config,
        best_val_loss=1.0,
        best_epoch=1,
        training_history={"train_loss": [1.0], "val_loss": [1.0]},
        path=path,
    )


def _save_tiny_test_pt(dataset: SpyDataset, path: Path) -> None:
    """Serialize a SpyDataset to a .pt file that from_pt_file() can load."""
    sequences = dataset._sequences.numpy()
    targets   = dataset._targets.numpy()
    SpyDataset.save_split({"sequences": sequences, "targets": targets}, path)


# ── compute_test_metrics ───────────────────────────────────────────────────────

class TestComputeTestMetrics:
    def test_returns_all_per_step_keys(self):
        config  = _make_tiny_config()
        model   = _make_tiny_model(config)
        dataset = _make_synthetic_dataset(64)

        metrics = compute_test_metrics(model, dataset)

        for step in range(1, DEFAULT_PRED_STEPS + 1):
            assert f"mae_step_{step}"                  in metrics
            assert f"rmse_step_{step}"                 in metrics
            assert f"r2_step_{step}"                   in metrics
            assert f"directional_accuracy_step_{step}" in metrics

    def test_returns_aggregate_keys(self):
        config  = _make_tiny_config()
        model   = _make_tiny_model(config)
        dataset = _make_synthetic_dataset(64)

        metrics = compute_test_metrics(model, dataset)

        assert "mae_aggregate"                  in metrics
        assert "rmse_aggregate"                 in metrics
        assert "r2_aggregate"                   in metrics
        assert "directional_accuracy_aggregate" in metrics

    def test_all_metric_values_are_finite(self):
        config  = _make_tiny_config()
        model   = _make_tiny_model(config)
        dataset = _make_synthetic_dataset(64)

        metrics = compute_test_metrics(model, dataset)

        assert all(np.isfinite(v) for v in metrics.values())

    def test_mae_is_non_negative(self):
        config  = _make_tiny_config()
        model   = _make_tiny_model(config)
        dataset = _make_synthetic_dataset(64)

        metrics = compute_test_metrics(model, dataset)

        for step in range(1, DEFAULT_PRED_STEPS + 1):
            assert metrics[f"mae_step_{step}"] >= 0.0
        assert metrics["mae_aggregate"] >= 0.0


# ── sample_latency_stats ───────────────────────────────────────────────────────

class TestSampleLatencyStats:
    def test_returns_required_keys(self):
        config = _make_tiny_config()
        model  = _make_tiny_model(config)

        stats = sample_latency_stats(model, config, n_samples=5)

        assert "latency_mean_ms"        in stats
        assert "latency_max_ms"         in stats
        assert "latency_violation_count" in stats
        assert "latency_n_samples"      in stats

    def test_n_samples_matches_requested(self):
        config = _make_tiny_config()
        model  = _make_tiny_model(config)

        stats = sample_latency_stats(model, config, n_samples=7)

        assert stats["latency_n_samples"] == 7

    def test_latency_values_are_positive(self):
        config = _make_tiny_config()
        model  = _make_tiny_model(config)

        stats = sample_latency_stats(model, config, n_samples=5)

        assert stats["latency_mean_ms"] > 0.0
        assert stats["latency_max_ms"]  > 0.0

    def test_max_latency_at_least_mean(self):
        config = _make_tiny_config()
        model  = _make_tiny_model(config)

        stats = sample_latency_stats(model, config, n_samples=10)

        assert stats["latency_max_ms"] >= stats["latency_mean_ms"]

    def test_violation_count_is_non_negative_int(self):
        config = _make_tiny_config()
        model  = _make_tiny_model(config)

        stats = sample_latency_stats(model, config, n_samples=5)

        assert isinstance(stats["latency_violation_count"], int)
        assert stats["latency_violation_count"] >= 0


# ── run_evaluation ─────────────────────────────────────────────────────────────

class TestRunEvaluation:
    def test_returns_metrics_and_latency_keys(self, tmp_path, monkeypatch):
        """run_evaluation must return a dict with both metric and latency keys."""
        log_path = tmp_path / "log.json"
        monkeypatch.setattr("src.utils.logger.EXPERIMENT_LOG_PATH", log_path)

        config          = _make_tiny_config()
        model           = _make_tiny_model(config)
        checkpoint_path = tmp_path / "best.pth"
        data_path       = tmp_path / "test.pt"

        _save_tiny_checkpoint(model, config, checkpoint_path)
        _save_tiny_test_pt(_make_synthetic_dataset(32), data_path)

        result = run_evaluation(checkpoint_path, data_path, n_latency_samples=5)

        # Per-step metrics
        for step in range(1, DEFAULT_PRED_STEPS + 1):
            assert f"mae_step_{step}" in result
        # Aggregate
        assert "mae_aggregate" in result
        # Latency
        assert "latency_mean_ms"          in result
        assert "latency_max_ms"           in result
        assert "latency_violation_count"  in result

    def test_two_runs_produce_two_log_entries(self, tmp_path, monkeypatch):
        """
        Two consecutive calls to run_evaluation must produce two separate
        entries in the experiment log — never overwrite or merge.
        """
        log_path = tmp_path / "log.json"
        monkeypatch.setattr("src.utils.logger.EXPERIMENT_LOG_PATH", log_path)

        config          = _make_tiny_config()
        model           = _make_tiny_model(config)
        checkpoint_path = tmp_path / "best.pth"
        data_path       = tmp_path / "test.pt"

        _save_tiny_checkpoint(model, config, checkpoint_path)
        _save_tiny_test_pt(_make_synthetic_dataset(32), data_path)

        run_evaluation(checkpoint_path, data_path, n_latency_samples=5)
        run_evaluation(checkpoint_path, data_path, n_latency_samples=5)

        with open(log_path) as f:
            records = json.load(f)

        assert len(records) == 2
        assert records[0]["experiment_name"] == "evaluation_run"
        assert records[1]["experiment_name"] == "evaluation_run"

    def test_log_entry_contains_required_fields(self, tmp_path, monkeypatch):
        """Each log entry must contain timestamp, experiment_name, config, metrics."""
        log_path = tmp_path / "log.json"
        monkeypatch.setattr("src.utils.logger.EXPERIMENT_LOG_PATH", log_path)

        config          = _make_tiny_config()
        model           = _make_tiny_model(config)
        checkpoint_path = tmp_path / "best.pth"
        data_path       = tmp_path / "test.pt"

        _save_tiny_checkpoint(model, config, checkpoint_path)
        _save_tiny_test_pt(_make_synthetic_dataset(32), data_path)

        run_evaluation(checkpoint_path, data_path, n_latency_samples=5)

        with open(log_path) as f:
            records = json.load(f)

        entry = records[0]
        assert "timestamp"       in entry
        assert "experiment_name" in entry
        assert "config"          in entry
        assert "metrics"         in entry
