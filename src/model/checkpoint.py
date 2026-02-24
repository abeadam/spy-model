"""
Save and load SpyPredictor checkpoints.

A checkpoint captures everything needed to resume training or run inference:
    - model state dict
    - TrainingConfig snapshot (n_features validated on load)
    - best validation loss and the epoch it occurred at
    - full per-epoch training history

The n_features field is validated on load to catch mismatches between the
checkpoint and the data being fed into the model.
"""

from pathlib import Path
from typing import Any

import torch

from src.utils.config import COMPUTE_DEVICE, DEFAULT_CONFIG, TrainingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    model: "SpyPredictor",  # noqa: F821  (forward reference, avoids circular import)
    config: TrainingConfig,
    best_val_loss: float,
    best_epoch: int,
    training_history: dict[str, list[float]],
    path: Path,
) -> None:
    """
    Save a checkpoint to `path`.

    Parameters
    ----------
    model : SpyPredictor
        The model whose weights should be saved.
    config : TrainingConfig
        Hyperparameter snapshot from this training run.
    best_val_loss : float
        Validation loss at the epoch being saved.
    best_epoch : int
        The epoch index (1-based) that produced this checkpoint.
    training_history : dict[str, list[float]]
        Per-epoch "train_loss" and "val_loss" lists up to best_epoch.
    path : Path
        Destination file path (typically models/spy_predictor_best.pth).
    """
    # torch.compile() wraps the model in an OptimizedModule whose state_dict()
    # has "_orig_mod." prefixed keys. Unwrap to always save bare keys so that
    # load_checkpoint can restore into a fresh (uncompiled) SpyPredictor.
    underlying_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict":  underlying_model.state_dict(),
            "config":            config.to_dict(),
            "best_val_loss":     best_val_loss,
            "best_epoch":        best_epoch,
            "training_history":  training_history,
            "n_features":        config.n_features,
        },
        path,
    )
    logger.info(
        "Checkpoint saved → %s  (epoch %d, val_loss=%.6f)",
        path, best_epoch, best_val_loss,
    )


def load_checkpoint(
    path: Path,
    device: torch.device = COMPUTE_DEVICE,
) -> tuple[Any, TrainingConfig, dict]:
    """
    Load a SpyPredictor checkpoint from `path`.

    Parameters
    ----------
    path : Path
        Path to the .pth checkpoint file.
    device : torch.device
        Device to load the model onto.

    Returns
    -------
    model : SpyPredictor
        Model loaded with the saved weights, in eval mode.
    config : TrainingConfig
        TrainingConfig reconstructed from the checkpoint's config snapshot.
    metadata : dict
        Contains best_val_loss, best_epoch, training_history.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    ValueError
        If the checkpoint's n_features does not match the reconstructed config,
        or if the file is missing required keys.
    """
    # Import here to avoid circular dependency (spy_predictor imports from config).
    from src.model.spy_predictor import SpyPredictor

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    required_keys = {"model_state_dict", "config", "best_val_loss", "best_epoch", "n_features"}
    missing_keys  = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(f"Checkpoint is missing required keys: {missing_keys}")

    config_dict = checkpoint["config"]
    config      = TrainingConfig(**{
        field: config_dict[field]
        for field in TrainingConfig.__dataclass_fields__
        if field in config_dict
    })

    checkpoint_n_features = checkpoint["n_features"]
    if checkpoint_n_features != config.n_features:
        raise ValueError(
            f"Checkpoint n_features ({checkpoint_n_features}) does not match "
            f"config n_features ({config.n_features}). "
            "The checkpoint was trained on a different feature set."
        )

    model = SpyPredictor(config=config, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metadata = {
        "best_val_loss":    checkpoint["best_val_loss"],
        "best_epoch":       checkpoint["best_epoch"],
        "training_history": checkpoint.get("training_history", {}),
    }

    logger.info(
        "Checkpoint loaded ← %s  (epoch %d, val_loss=%.6f)",
        path, metadata["best_epoch"], metadata["best_val_loss"],
    )
    return model, config, metadata
