"""
PyTorch Dataset wrapping preprocessed SPY prediction sequences.

SpyDataset loads one split (train / val / test) and returns
(features_tensor, targets_tensor) pairs for use with DataLoader.

split_sequences_by_day() performs the chronological 80/10/10 split at the
trading-day level before flattening to sequences, which prevents any day's
data from leaking across split boundaries.
"""

from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import DEFAULT_CONFIG, TrainingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_sequences_by_day(
    all_sequences:  np.ndarray,
    all_targets:    np.ndarray,
    sequence_dates: list[date],
    config: TrainingConfig = DEFAULT_CONFIG,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Chronologically split sequences into train / val / test by trading day.

    The split boundary is computed on unique trading days, not on individual
    sequences. This ensures no day appears in more than one split and that
    the model never sees future data during training.

    Parameters
    ----------
    all_sequences : np.ndarray
        Shape (N, sequence_length, n_features).
    all_targets : np.ndarray
        Shape (N, prediction_steps).
    sequence_dates : list[date]
        One trading date per sequence (len == N).
    config : TrainingConfig
        Supplies train_split_ratio and validation_split_ratio.

    Returns
    -------
    dict with keys "train", "val", "test", each mapping to:
        {"sequences": np.ndarray, "targets": np.ndarray}
    """
    unique_dates = sorted(set(sequence_dates))
    total_days   = len(unique_dates)

    train_day_count = int(total_days * config.train_split_ratio)
    val_day_count   = int(total_days * config.validation_split_ratio)

    train_days = set(unique_dates[:train_day_count])
    val_days   = set(unique_dates[train_day_count : train_day_count + val_day_count])
    test_days  = set(unique_dates[train_day_count + val_day_count :])

    splits: dict[str, dict[str, np.ndarray]] = {}

    for split_name, day_set in [
        ("train", train_days),
        ("val",   val_days),
        ("test",  test_days),
    ]:
        mask = np.array([trading_date in day_set for trading_date in sequence_dates])
        splits[split_name] = {
            "sequences": all_sequences[mask],
            "targets":   all_targets[mask],
        }
        logger.info(
            "Split '%s': %d sequences from %d days.",
            split_name,
            int(mask.sum()),
            len(day_set),
        )

    return splits


class SpyDataset(Dataset):
    """
    PyTorch Dataset for one split of preprocessed SPY prediction sequences.

    Each item is a (features_tensor, targets_tensor) pair:
        features_tensor : torch.Tensor  shape (sequence_length, n_features)
        targets_tensor  : torch.Tensor  shape (prediction_steps,)

    Parameters
    ----------
    sequences : np.ndarray  shape (N, sequence_length, n_features)
    targets   : np.ndarray  shape (N, prediction_steps)
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        if len(sequences) != len(targets):
            raise ValueError(
                f"sequences and targets must have the same length. "
                f"Got {len(sequences)} vs {len(targets)}."
            )
        self._sequences = torch.from_numpy(sequences)
        self._targets   = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._sequences[index], self._targets[index]

    @classmethod
    def from_pt_file(cls, file_path: Path) -> "SpyDataset":
        """Load a dataset split from a pre-saved .pt file."""
        checkpoint = torch.load(file_path, weights_only=True)
        sequences  = checkpoint["sequences"]
        targets    = checkpoint["targets"]
        # Accept both Tensor and ndarray stored in the checkpoint.
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
        return cls(sequences=sequences, targets=targets)

    @staticmethod
    def save_split(split_data: dict[str, np.ndarray], file_path: Path) -> None:
        """Save a split dict (sequences + targets ndarrays) as a .pt file."""
        torch.save(
            {
                "sequences": torch.from_numpy(split_data["sequences"]),
                "targets":   torch.from_numpy(split_data["targets"]),
            },
            file_path,
        )
        logger.info(
            "Saved %d sequences to %s.", len(split_data["sequences"]), file_path
        )
