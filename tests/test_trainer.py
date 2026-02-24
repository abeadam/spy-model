"""
Tests for src/training/trainer.py

Uses a tiny model and tiny synthetic dataset to keep tests fast (< 5s each).
All tests use CPU so MPS availability doesn't affect results.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import SpyDataset
from src.model.checkpoint import load_checkpoint
from src.model.spy_predictor import SpyPredictor
from src.training.trainer import Trainer
from src.utils.config import TrainingConfig


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_tiny_config(
    max_epochs: int = 20,
    early_stopping_patience: int = 3,
    batch_size: int = 32,
) -> TrainingConfig:
    """Tiny config for fast CPU tests."""
    return TrainingConfig(
        d_model=16,
        n_heads=2,
        n_encoder_layers=1,
        ffn_dim=32,
        dropout_rate=0.0,
        learning_rate=1e-3,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        lr_scheduler_t0=10,
        random_seed=42,
        # Data dimensions must match the dataset.
        n_features=DEFAULT_N_FEATURES,
        sequence_length=DEFAULT_SEQ_LEN,
        prediction_steps=DEFAULT_PRED_STEPS,
    )


DEFAULT_N_FEATURES  = 26
DEFAULT_SEQ_LEN     = 120
DEFAULT_PRED_STEPS  = 5


def _make_synthetic_dataset(
    n_samples: int,
    seed: int = 0,
    target_mean: float = 500.0,
) -> SpyDataset:
    """Build a synthetic dataset with random normalized inputs and fixed-range targets."""
    rng       = np.random.default_rng(seed=seed)
    sequences = rng.standard_normal((n_samples, DEFAULT_SEQ_LEN, DEFAULT_N_FEATURES)).astype(np.float32)
    targets   = rng.uniform(target_mean - 5.0, target_mean + 5.0, (n_samples, DEFAULT_PRED_STEPS)).astype(np.float32)
    return SpyDataset(sequences, targets)


def _make_constant_target_dataset(
    n_samples: int,
    target_value: float = 500.0,
    seed: int = 0,
) -> SpyDataset:
    """
    Dataset with random inputs but constant targets.

    Because every target is the same scalar, the model converges in a few
    epochs and then val loss plateaus — reliably triggering early stopping.
    """
    rng       = np.random.default_rng(seed=seed)
    sequences = rng.standard_normal((n_samples, DEFAULT_SEQ_LEN, DEFAULT_N_FEATURES)).astype(np.float32)
    targets   = np.full((n_samples, DEFAULT_PRED_STEPS), target_value, dtype=np.float32)
    return SpyDataset(sequences, targets)


def _make_trainer(
    config: TrainingConfig,
    checkpoint_path: Path,
    n_train: int = 128,
    n_val: int = 64,
) -> Trainer:
    model         = SpyPredictor(config=config, device=torch.device("cpu"))
    train_dataset = _make_synthetic_dataset(n_train, seed=1)
    val_dataset   = _make_synthetic_dataset(n_val,   seed=2)
    return Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_path=checkpoint_path,
    )


# ── Training history ──────────────────────────────────────────────────────────

class TestTrainerHistory:
    def test_history_has_train_and_val_loss_keys(self, tmp_path):
        config  = _make_tiny_config(max_epochs=3, early_stopping_patience=10)
        trainer = _make_trainer(config, tmp_path / "ckpt.pth")
        history = trainer.train()
        assert "train_loss" in history
        assert "val_loss"   in history

    def test_history_length_matches_epochs_run(self, tmp_path):
        config  = _make_tiny_config(max_epochs=3, early_stopping_patience=10)
        trainer = _make_trainer(config, tmp_path / "ckpt.pth")
        history = trainer.train()
        assert len(history["train_loss"]) == len(history["val_loss"])
        assert len(history["train_loss"]) <= config.max_epochs

    def test_all_losses_are_positive(self, tmp_path):
        config  = _make_tiny_config(max_epochs=3, early_stopping_patience=10)
        trainer = _make_trainer(config, tmp_path / "ckpt.pth")
        history = trainer.train()
        assert all(loss > 0.0 for loss in history["train_loss"])
        assert all(loss > 0.0 for loss in history["val_loss"])


# ── Early stopping ────────────────────────────────────────────────────────────

def _make_mock_run_epoch(controlled_val_losses: list[float]):
    """
    Return a mock for Trainer._run_epoch that replaces real forward passes.

    train epochs always return 0.1 (shrinking), val epochs cycle through
    the controlled sequence so we can test exact early-stopping behaviour.
    """
    val_call_count = {"n": 0}

    def mock_run_epoch(optimizer, loss_fn, training: bool) -> float:  # no `self`
        if training:
            return 0.1
        loss = controlled_val_losses[val_call_count["n"]]
        val_call_count["n"] += 1
        return loss

    return mock_run_epoch


class TestEarlyStopping:
    def test_training_stops_before_max_epochs(self, tmp_path, monkeypatch):
        """
        Inject a controlled val-loss sequence (5 improving, then plateau) and
        verify training stops well before max_epochs.
        """
        patience = 3
        config   = _make_tiny_config(max_epochs=50, early_stopping_patience=patience)
        trainer  = _make_trainer(config, tmp_path / "ckpt.pth")

        # Val losses decrease for 5 epochs, then plateau — early stop at epoch 5+3=8.
        val_losses = [1.0, 0.9, 0.8, 0.7, 0.6] + [0.65] * 50
        monkeypatch.setattr(trainer, "_run_epoch", _make_mock_run_epoch(val_losses))

        history = trainer.train()
        assert len(history["val_loss"]) < config.max_epochs

    def test_epochs_run_equals_best_epoch_plus_patience(self, tmp_path, monkeypatch):
        """
        With a controlled val-loss sequence, the total epoch count must equal
        the best epoch index + patience (the exact early-stop point).
        """
        patience = 3
        n_improving = 5
        config  = _make_tiny_config(max_epochs=50, early_stopping_patience=patience)
        trainer = _make_trainer(config, tmp_path / "ckpt.pth")

        val_losses = list(np.linspace(1.0, 0.5, n_improving)) + [0.6] * 50
        monkeypatch.setattr(trainer, "_run_epoch", _make_mock_run_epoch(val_losses))

        history    = trainer.train()
        epochs_run = len(history["val_loss"])
        best_index = int(np.argmin(history["val_loss"]))

        assert epochs_run == best_index + 1 + patience


# ── Best checkpoint ───────────────────────────────────────────────────────────

class TestBestCheckpoint:
    def test_checkpoint_is_created(self, tmp_path):
        config          = _make_tiny_config(max_epochs=3, early_stopping_patience=10)
        checkpoint_path = tmp_path / "best.pth"
        trainer         = _make_trainer(config, checkpoint_path)
        trainer.train()
        assert checkpoint_path.exists()

    def test_checkpoint_val_loss_equals_minimum_val_loss(self, tmp_path):
        """
        The checkpoint must capture the epoch with the lowest val loss,
        not the final epoch.
        """
        config          = _make_tiny_config(max_epochs=10, early_stopping_patience=10)
        checkpoint_path = tmp_path / "best.pth"
        trainer         = _make_trainer(config, checkpoint_path)
        history         = trainer.train()

        _, _, metadata  = load_checkpoint(checkpoint_path, device=torch.device("cpu"))
        minimum_val_loss = min(history["val_loss"])

        assert metadata["best_val_loss"] == pytest.approx(minimum_val_loss, rel=1e-5)

    def test_checkpoint_best_epoch_matches_history(self, tmp_path):
        """best_epoch in the checkpoint must correspond to the argmin of val_loss."""
        config          = _make_tiny_config(max_epochs=10, early_stopping_patience=10)
        checkpoint_path = tmp_path / "best.pth"
        trainer         = _make_trainer(config, checkpoint_path)
        history         = trainer.train()

        _, _, metadata  = load_checkpoint(checkpoint_path, device=torch.device("cpu"))
        best_index      = int(np.argmin(history["val_loss"]))
        # best_epoch is 1-based in the checkpoint.
        assert metadata["best_epoch"] == best_index + 1

    def test_loaded_model_produces_valid_output(self, tmp_path):
        """A model loaded from the checkpoint must produce finite predictions."""
        config          = _make_tiny_config(max_epochs=3, early_stopping_patience=10)
        checkpoint_path = tmp_path / "best.pth"
        trainer         = _make_trainer(config, checkpoint_path)
        trainer.train()

        model, _, _ = load_checkpoint(checkpoint_path, device=torch.device("cpu"))
        rng     = np.random.default_rng(seed=99)
        raw_win = rng.uniform(490, 510, (DEFAULT_SEQ_LEN, DEFAULT_N_FEATURES)).astype(np.float32)
        prices, signal, _ = model.predict(raw_win)

        assert np.isfinite(prices).all()
        assert signal in {"up", "down"}


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_seed_produces_identical_training_history(self, tmp_path):
        """
        Two training runs with identical seed, config, and data must produce
        the same per-epoch loss history. The seed is set before each model
        instantiation so weight initialization is also identical.
        """
        config = _make_tiny_config(max_epochs=5, early_stopping_patience=10)

        # Seed before creating each model to ensure identical weight init.
        torch.manual_seed(config.random_seed)
        trainer_a = _make_trainer(config, tmp_path / "a.pth", n_train=128, n_val=64)
        history_a = trainer_a.train()

        torch.manual_seed(config.random_seed)
        trainer_b = _make_trainer(config, tmp_path / "b.pth", n_train=128, n_val=64)
        history_b = trainer_b.train()

        assert np.allclose(history_a["train_loss"], history_b["train_loss"], rtol=1e-4)
        assert np.allclose(history_a["val_loss"],   history_b["val_loss"],   rtol=1e-4)
