"""
Central configuration for all hyperparameters, paths, and training settings.
All other modules import from here — nothing is hardcoded elsewhere.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR        = PROJECT_ROOT / "results"
MODELS_DIR         = PROJECT_ROOT / "models"

EXPERIMENT_LOG_PATH = RESULTS_DIR / "experiment_log.json"

# Create directories if they don't exist
for directory in (DATA_RAW_DIR, DATA_PROCESSED_DIR, RESULTS_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ── Training configuration ────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Data
    sequence_length: int = 60          # Number of time steps fed into the model
    train_split_ratio: float = 0.8     # Fraction of data used for training
    validation_split_ratio: float = 0.1

    # Model
    hidden_size: int = 256
    num_layers: int = 4
    dropout_rate: float = 0.2

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 500
    early_stopping_patience: int = 20  # Stop if val loss doesn't improve

    # Evaluation
    random_seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = TrainingConfig()
