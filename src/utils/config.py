"""
Central configuration for all hyperparameters, paths, and training settings.
All other modules import from here — nothing is hardcoded elsewhere.
"""

import torch
from dataclasses import dataclass, asdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# data/raw/daily_data is a symlink → /Users/abeadam/dev/interactive-broker-python/Updated Stats/daily_data
# Set it up once with:
#   ln -sfn "/Users/abeadam/dev/interactive-broker-python/Updated Stats/daily_data" data/raw/daily_data
# Code always reads through the symlink — the source directory is never referenced directly.
DATA_RAW_DIR        = PROJECT_ROOT / "data" / "raw" / "daily_data"
DATA_PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR         = PROJECT_ROOT / "results"
MODELS_DIR          = PROJECT_ROOT / "models"

EXPERIMENT_LOG_PATH    = RESULTS_DIR / "experiment_log.json"
BEST_CHECKPOINT_PATH   = MODELS_DIR / "spy_predictor_best.pth"

# DATA_RAW_DIR is intentionally excluded — it must be a symlink, never auto-created.
for directory in (DATA_PROCESSED_DIR, RESULTS_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ── Device ───────────────────────────────────────────────────────────────────

def resolve_compute_device() -> torch.device:
    """
    Select the best available compute device.
    Prefers MPS (Apple Metal) for Apple Silicon Macs, then CUDA, then CPU.
    Resolved once at startup and shared across all modules.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


COMPUTE_DEVICE = resolve_compute_device()


# ── Model and training configuration ─────────────────────────────────────────

@dataclass
class TrainingConfig:
    # ── Data
    sequence_length: int = 120         # Lookback window in bars
    prediction_steps: int = 5          # Number of future bars to predict (25 seconds)
    n_features: int = 26               # 26 features: SPY/VIX/EMA + RSI(3) + momentum + BB(6) + ATR(3) + MFI(3) + RealVol(3)
    train_split_ratio: float = 0.80    # Fraction of trading days used for training
    validation_split_ratio: float = 0.10
    test_split_ratio: float = 0.10

    # ── Model architecture
    d_model: int = 128                 # Embedding / hidden dimension
    n_heads: int = 4                   # Attention heads (head_dim = d_model / n_heads = 32)
    n_encoder_layers: int = 3          # Number of transformer encoder blocks
    ffn_dim: int = 512                 # Feed-forward network inner dimension (4 × d_model)
    dropout_rate: float = 0.2

    # ── Training
    learning_rate: float = 3e-4        # AdamW starting learning rate
    batch_size: int = 256
    max_epochs: int = 1000             # Unconstrained — early stopping governs
    early_stopping_patience: int = 30  # Epochs of no val-loss improvement before stopping
    lr_scheduler_t0: int = 50          # CosineAnnealingWarmRestarts: epochs in first restart cycle
    random_seed: int = 42
    huber_loss_delta: float = 1.0      # Huber loss threshold: L2 below delta, L1 above

    # ── Indicators (periods in bars)
    ema_short_period: int = 5
    ema_mid_period: int = 9
    ema_long_period: int = 12
    rsi_period: int = 7
    momentum_period: int = 5           # Used for both price and volume momentum
    indicator_periods: tuple = (7, 14, 28)  # Periods for RSI, BB, ATR, MFI, realized vol

    # ── Target transform
    # Controls how sequence_builder.py encodes the 5-bar future closes as targets.
    # Options: "percent_change" | "basis_points" | "signed_log" |
    #          "vol_normalized" | "vol_normalized_log" | "vol_normalized_log_prevday"
    target_transform: str = "percent_change"

    # ── Evaluation
    max_inference_latency_ms: float = 100.0  # Violations above this are flagged in the log

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = TrainingConfig()
