"""
Learning rate scheduler for SpyPredictor training.

CosineAnnealingWarmRestarts periodically resets the LR to its initial value,
which helps the optimizer escape local minima and explore the loss landscape.
Restarts happen every T_0 epochs; eta_min sets the floor LR between restarts.
"""

import torch

from src.utils.config import DEFAULT_CONFIG, TrainingConfig


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig = DEFAULT_CONFIG,
) -> torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
    """
    Build the learning rate scheduler from config.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose LR will be scheduled.
    config : TrainingConfig
        Supplies lr_scheduler_t0 (epochs per restart cycle).

    Returns
    -------
    CosineAnnealingWarmRestarts
        Call scheduler.step() once per epoch after the optimizer step.
    """
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.lr_scheduler_t0,
        T_mult=1,
        eta_min=1e-6,
    )
