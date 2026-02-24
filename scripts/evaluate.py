#!/usr/bin/env python
"""
Evaluation entry point: load a saved checkpoint → run inference on the test set
→ report per-step and aggregate metrics → report latency statistics → log result.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint models/spy_predictor_best.pth
    python scripts/evaluate.py --checkpoint models/spy_predictor_best.pth \
                               --data data/processed/test.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import SpyDataset
from src.evaluation.latency_tracker import measure_inference_latency
from src.evaluation.metrics import compute_per_step_metrics
from src.model.checkpoint import load_checkpoint
from src.model.spy_predictor import SpyPredictor
from src.utils.config import (
    BEST_CHECKPOINT_PATH,
    DATA_PROCESSED_DIR,
    DEFAULT_CONFIG,
    TrainingConfig,
)
from src.utils.logger import get_logger, save_experiment_result

logger = get_logger("evaluate")

# Number of synthetic windows used to estimate inference latency.
# Enough to capture warm-up variability; not so many that evaluation is slow.
_N_LATENCY_SAMPLES = 50

# Price range for synthetic latency windows. Chosen to match typical SPY close
# prices so the per-feature normalization path sees realistic variance.
_SYNTHETIC_PRICE_LOW  = 490.0
_SYNTHETIC_PRICE_HIGH = 510.0

# Fixed seed for deterministic latency sampling across runs.
_LATENCY_SAMPLING_SEED = 0


def compute_test_metrics(
    model: SpyPredictor,
    test_dataset: SpyDataset,
) -> dict[str, float]:
    """
    Batched inference over the test set; returns per-step + aggregate metrics.

    Uses model() directly (not predict()) for throughput over large datasets.
    The test dataset stores already-normalized sequences that match the model's
    expected input format.
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

    predicted_array = np.concatenate(all_predictions, axis=0)  # (N, prediction_steps)
    actual_array    = np.concatenate(all_targets,     axis=0)  # (N, prediction_steps)

    return compute_per_step_metrics(actual_array, predicted_array)


def sample_latency_stats(
    model: SpyPredictor,
    config: TrainingConfig,
    n_samples: int = _N_LATENCY_SAMPLES,
) -> dict[str, float]:
    """
    Estimate inference latency by running predict() on synthetic windows.

    Synthetic windows use uniform random prices in [490, 510] which produce
    well-conditioned per-feature z-scores — the same normalization path the
    model uses in production.

    Returns a dict with keys:
        latency_mean_ms, latency_max_ms, latency_violation_count, latency_n_samples
    """
    rng                = np.random.default_rng(seed=_LATENCY_SAMPLING_SEED)
    latency_samples    = np.empty(n_samples, dtype=np.float64)
    violation_count    = 0

    for i in range(n_samples):
        raw_window = rng.uniform(
            _SYNTHETIC_PRICE_LOW,
            _SYNTHETIC_PRICE_HIGH,
            (config.sequence_length, config.n_features),
        ).astype(np.float32)
        _, outer_latency_ms, is_violation = measure_inference_latency(
            model.predict, raw_window, config
        )
        latency_samples[i] = outer_latency_ms
        if is_violation:
            violation_count += 1

    return {
        "latency_mean_ms":       float(np.mean(latency_samples)),
        "latency_max_ms":        float(np.max(latency_samples)),
        "latency_violation_count": violation_count,
        "latency_n_samples":     n_samples,
    }


def run_evaluation(
    checkpoint_path: Path,
    data_path: Path,
    n_latency_samples: int = _N_LATENCY_SAMPLES,
    use_compile: bool = False,
) -> dict[str, float]:
    """
    Full evaluation pipeline: load checkpoint → compile (optional) →
    compute test metrics → sample latency → log result.

    Parameters
    ----------
    checkpoint_path : Path
        Saved .pth checkpoint to evaluate.
    data_path : Path
        Test dataset .pt file.
    n_latency_samples : int
        Number of synthetic windows used to estimate inference latency.
    use_compile : bool
        If True and torch.compile is available, compile the model with
        mode="reduce-overhead" before inference. Disabled by default so
        unit tests avoid JIT warm-up overhead.

    Returns the combined metrics + latency dict that was written to the log.
    """
    logger.info("Loading checkpoint: %s", checkpoint_path)
    model, config, checkpoint_metadata = load_checkpoint(
        checkpoint_path, device=torch.device("cpu")
    )
    logger.info(
        "Checkpoint loaded — best_epoch=%d, best_val_loss=%.6f",
        checkpoint_metadata["best_epoch"],
        checkpoint_metadata["best_val_loss"],
    )

    if use_compile:
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("torch.compile() enabled (mode=reduce-overhead).")
        else:
            logger.info("torch.compile() not available; running in eager mode.")

    logger.info("Loading test dataset: %s", data_path)
    test_dataset = SpyDataset.from_pt_file(data_path)
    logger.info("Test dataset: %d samples", len(test_dataset))

    logger.info("Computing per-step metrics on test set ...")
    test_metrics = compute_test_metrics(model, test_dataset)

    prediction_steps = config.prediction_steps
    for step in range(1, prediction_steps + 1):
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

    logger.info("Sampling inference latency (%d windows) ...", n_latency_samples)
    latency_stats = sample_latency_stats(model, config, n_latency_samples)
    logger.info(
        "  Latency — mean=%.2f ms, max=%.2f ms, violations=%d/%d",
        latency_stats["latency_mean_ms"],
        latency_stats["latency_max_ms"],
        latency_stats["latency_violation_count"],
        latency_stats["latency_n_samples"],
    )

    combined_metrics = {**test_metrics, **latency_stats}

    save_experiment_result(
        experiment_name="evaluation_run",
        config=config.to_dict(),
        metrics=combined_metrics,
        notes=(
            f"checkpoint={checkpoint_path}, "
            f"best_epoch={checkpoint_metadata['best_epoch']}, "
            f"best_val_loss={checkpoint_metadata['best_val_loss']:.6f}"
        ),
    )
    logger.info("Result appended to experiment log.")

    return combined_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SPY predictor checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BEST_CHECKPOINT_PATH,
        metavar="CHECKPOINT",
        help="Path to the .pth checkpoint to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PROCESSED_DIR / "test.pt",
        metavar="DATA",
        help="Path to the test .pt dataset file (default: %(default)s).",
    )
    args = parser.parse_args()

    logger.info("=== SPY Predictor Evaluation ===")
    logger.info("Checkpoint : %s", args.checkpoint)
    logger.info("Data       : %s", args.data)
    run_evaluation(args.checkpoint, args.data, use_compile=True)
    logger.info("=== Evaluation complete. ===")


if __name__ == "__main__":
    main()
