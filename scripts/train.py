#!/usr/bin/env python
"""
Training entry point: load processed data → train → evaluate on test set → log.

Usage:
    /Users/abeadam/dev/model/Stock-Model/.venv/bin/python scripts/train.py
    /Users/abeadam/dev/model/Stock-Model/.venv/bin/python scripts/train.py --resume models/spy_predictor_best.pth
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import SpyDataset
from src.evaluation.metrics import compute_per_step_metrics
from src.model.checkpoint import load_checkpoint
from src.model.spy_predictor import SpyPredictor
from src.training.trainer import Trainer
from src.utils.config import (
    BEST_CHECKPOINT_PATH,
    COMPUTE_DEVICE,
    DATA_PROCESSED_DIR,
    DEFAULT_CONFIG,
    TrainingConfig,
)
from src.data.sequence_builder import _VALID_TRANSFORMS
from src.utils.logger import get_logger, save_experiment_result

logger = get_logger("train")


def evaluate_on_test_set(
    model: SpyPredictor,
    test_dataset: SpyDataset,
) -> dict[str, float]:
    """
    Run batched inference over the test set and return per-step + aggregate metrics.

    Uses the model's forward pass directly (not predict()) for efficiency
    over 250K+ test samples.
    """
    config     = model.config
    dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_predictions: list[np.ndarray] = []
    all_targets:     list[np.ndarray] = []

    model.eval()
    with torch.inference_mode():
        for features, targets in dataloader:
            features    = features.to(model._compute_device)
            predictions = model(features).cpu().numpy()
            all_predictions.append(predictions)
            all_targets.append(targets.numpy())

    predicted_array = np.concatenate(all_predictions, axis=0)  # (N, 5)
    actual_array    = np.concatenate(all_targets,     axis=0)  # (N, 5)

    return compute_per_step_metrics(actual_array, predicted_array)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SPY predictor model.")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help="Path to a .pth checkpoint to resume training from.",
    )
    parser.add_argument(
        "--target-transform",
        default=DEFAULT_CONFIG.target_transform,
        choices=sorted(_VALID_TRANSFORMS),
        help="Target encoding that matches the preprocessed .pt files (default: %(default)s).",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=None,
        metavar="DELTA",
        help="Huber loss delta; overrides config.huber_loss_delta if provided.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_PROCESSED_DIR,
        metavar="DIR",
        help="Directory containing train.pt, val.pt, test.pt (default: data/processed/).",
    )
    args = parser.parse_args()

    config = TrainingConfig(target_transform=args.target_transform)
    if args.huber_delta is not None:
        config = dataclasses.replace(config, huber_loss_delta=args.huber_delta)
    logger.info("=== SPY Predictor Training ===")
    logger.info("Compute device : %s", COMPUTE_DEVICE)
    logger.info("Checkpoint     : %s", BEST_CHECKPOINT_PATH)

    # Load preprocessed data splits.
    data_dir = args.data_dir
    logger.info("Loading preprocessed data from %s ...", data_dir)
    train_dataset = SpyDataset.from_pt_file(data_dir / "train.pt")
    val_dataset   = SpyDataset.from_pt_file(data_dir / "val.pt")
    test_dataset  = SpyDataset.from_pt_file(data_dir / "test.pt")
    logger.info(
        "Dataset sizes — train: %d, val: %d, test: %d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )

    # Instantiate model.
    model = SpyPredictor(config=config, device=COMPUTE_DEVICE)
    logger.info("Model parameters: %d", model.parameter_count())

    # torch.compile() speeds up training on PyTorch 2.0+ by fusing operations.
    # Skipped gracefully on older versions that don't expose the API.
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("torch.compile() enabled (mode=reduce-overhead).")
    else:
        logger.info("torch.compile() not available; running in eager mode.")

    # Train.
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_path=BEST_CHECKPOINT_PATH,
    )
    training_history = trainer.train(resume_checkpoint_path=args.resume)

    # Evaluate best checkpoint on the test set.
    logger.info("Evaluating best checkpoint on test set ...")
    best_model, _, checkpoint_metadata = load_checkpoint(
        BEST_CHECKPOINT_PATH, device=COMPUTE_DEVICE
    )
    test_metrics = evaluate_on_test_set(best_model, test_dataset)

    logger.info("Test metrics:")
    for step in range(1, config.prediction_steps + 1):
        logger.info(
            "  Step %d — MAE=%.4f  RMSE=%.4f  R²=%.4f  DirAcc=%.1f%%",
            step,
            test_metrics[f"mae_step_{step}"],
            test_metrics[f"rmse_step_{step}"],
            test_metrics[f"r2_step_{step}"],
            test_metrics[f"directional_accuracy_step_{step}"],
        )
    logger.info(
        "  Aggregate — MAE=%.4f  RMSE=%.4f  R²=%.4f  DirAcc=%.1f%%",
        test_metrics["mae_aggregate"],
        test_metrics["rmse_aggregate"],
        test_metrics["r2_aggregate"],
        test_metrics["directional_accuracy_aggregate"],
    )

    # Append full result to experiment log.
    save_experiment_result(
        experiment_name="training_run",
        config=config.to_dict(),
        metrics=test_metrics,
        notes=(
            f"best_epoch={checkpoint_metadata['best_epoch']}, "
            f"best_val_loss={checkpoint_metadata['best_val_loss']:.6f}, "
            f"epochs_run={len(training_history['train_loss'])}"
        ),
    )
    logger.info("Result appended to experiment log.")
    logger.info("=== Training complete. ===")


if __name__ == "__main__":
    main()
