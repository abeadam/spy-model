"""
Training loop for SpyPredictor with early stopping and best-checkpoint saving.

The Trainer:
    - Wraps SpyDataset instances into DataLoaders
    - Trains with AdamW optimizer and Huber loss
    - Applies CosineAnnealingWarmRestarts LR scheduling (once per epoch)
    - Tracks per-epoch train and val loss
    - Saves the best checkpoint whenever val loss improves
    - Stops early after `early_stopping_patience` epochs with no improvement
    - Supports resuming from a saved checkpoint (loads weights + prior best val loss)
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import SpyDataset
from src.model.checkpoint import load_checkpoint, save_checkpoint
from src.model.spy_predictor import SpyPredictor
from src.training.scheduler import build_lr_scheduler
from src.utils.config import BEST_CHECKPOINT_PATH, DEFAULT_CONFIG, TrainingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Manages the full training lifecycle for SpyPredictor.

    Parameters
    ----------
    model : SpyPredictor
        Un-trained (or pre-trained, for resume) model instance.
    config : TrainingConfig
        All training hyperparameters.
    train_dataset : SpyDataset
        Training split.
    val_dataset : SpyDataset
        Validation split used for early stopping and checkpoint decisions.
    checkpoint_path : Path
        Where to write the best checkpoint file.
    """

    def __init__(
        self,
        model: SpyPredictor,
        config: TrainingConfig,
        train_dataset: SpyDataset,
        val_dataset: SpyDataset,
        checkpoint_path: Path = BEST_CHECKPOINT_PATH,
    ) -> None:
        self._model           = model
        self._config          = config
        self._checkpoint_path = checkpoint_path

        # num_workers=0 avoids multiprocessing issues on macOS/MPS.
        # Data is already in memory from .pt files, so IO is not the bottleneck.
        self._train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self._val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def train(
        self,
        resume_checkpoint_path: Optional[Path] = None,
    ) -> dict[str, list[float]]:
        """
        Run the training loop until early stopping or max_epochs is reached.

        Parameters
        ----------
        resume_checkpoint_path : Path, optional
            If provided, load this checkpoint's weights and best_val_loss before
            starting training. The optimizer is always initialized fresh.

        Returns
        -------
        dict[str, list[float]]
            Training history with keys "train_loss" and "val_loss", one entry
            per epoch that ran.
        """
        torch.manual_seed(self._config.random_seed)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._config.learning_rate,
        )
        scheduler = build_lr_scheduler(optimizer, self._config)
        loss_fn   = nn.HuberLoss(delta=self._config.huber_loss_delta, reduction="mean")

        best_val_loss:    float                    = float("inf")
        patience_counter: int                      = 0
        training_history: dict[str, list[float]]  = {"train_loss": [], "val_loss": []}

        if resume_checkpoint_path is not None:
            _, _, metadata = load_checkpoint(
                resume_checkpoint_path, device=self._model._compute_device
            )
            best_val_loss    = metadata["best_val_loss"]
            training_history = metadata["training_history"]
            logger.info(
                "Resumed from checkpoint. Best val loss so far: %.6f", best_val_loss
            )

        logger.info(
            "Starting training — max_epochs=%d, patience=%d, device=%s",
            self._config.max_epochs,
            self._config.early_stopping_patience,
            self._model._compute_device,
        )

        for epoch in range(1, self._config.max_epochs + 1):
            train_loss = self._run_epoch(optimizer, loss_fn, training=True)
            val_loss   = self._run_epoch(None,      loss_fn, training=False)

            scheduler.step()

            training_history["train_loss"].append(train_loss)
            training_history["val_loss"].append(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %4d | train_loss=%.6f | val_loss=%.6f | lr=%.2e",
                epoch, train_loss, val_loss, current_lr,
            )

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                save_checkpoint(
                    model=self._model,
                    config=self._config,
                    best_val_loss=best_val_loss,
                    best_epoch=epoch,
                    training_history=training_history,
                    path=self._checkpoint_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= self._config.early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d — no improvement for %d epochs.",
                        epoch,
                        self._config.early_stopping_patience,
                    )
                    break

        logger.info(
            "Training complete. Best val loss: %.6f", best_val_loss
        )
        return training_history

    def _run_epoch(
        self,
        optimizer: Optional[torch.optim.Optimizer],
        loss_fn: nn.Module,
        training: bool,
    ) -> float:
        """
        Run one pass (train or validation) over the respective DataLoader.

        Returns the average Huber loss across all samples in this epoch.
        """
        if training:
            self._model.train()
        else:
            self._model.eval()

        accumulated_loss    = 0.0
        total_samples_seen  = 0

        if training:
            for features, targets in self._train_loader:
                features = features.to(self._model._compute_device)
                targets  = targets.to(self._model._compute_device)

                optimizer.zero_grad(set_to_none=True)
                predictions = self._model(features)
                loss        = loss_fn(predictions, targets)
                loss.backward()
                optimizer.step()

                batch_size           = features.size(0)
                accumulated_loss    += loss.item() * batch_size
                total_samples_seen  += batch_size
        else:
            with torch.inference_mode():
                for features, targets in self._val_loader:
                    features = features.to(self._model._compute_device)
                    targets  = targets.to(self._model._compute_device)

                    predictions = self._model(features)
                    loss        = loss_fn(predictions, targets)

                    batch_size          = features.size(0)
                    accumulated_loss   += loss.item() * batch_size
                    total_samples_seen += batch_size

        return accumulated_loss / total_samples_seen
