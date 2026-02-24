#!/usr/bin/env python
"""
One-time preprocessing pipeline: raw bar files → processed .pt tensors.

Outputs three files to data/processed/:
    train.pt  — training split (earliest 80% of trading days)
    val.pt    — validation split (next 10% of trading days)
    test.pt   — test split (final 10% of trading days)

Each .pt file is a dict with keys:
    "sequences"  torch.Tensor  shape (N, 60, 9)  float32  z-score normalized
    "targets"    torch.Tensor  shape (N, 5)       float32  raw SPY dollar prices

Run from the project root:
    /Users/abeadam/dev/model/Stock-Model/.venv/bin/python scripts/preprocess.py
"""

import argparse
import sys
from pathlib import Path

# Allow top-level imports from the project root (src/ package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bar_loader import load_aligned_daily_bars
from src.data.indicator_computer import compute_indicators
from src.data.sequence_builder import build_all_sequences, _VALID_TRANSFORMS
from src.data.dataset import SpyDataset, split_sequences_by_day
from src.utils.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, DEFAULT_CONFIG, TrainingConfig
from src.utils.logger import get_logger

logger = get_logger("preprocess")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess SPY bar data into .pt tensors.")
    parser.add_argument(
        "--target-transform",
        default=DEFAULT_CONFIG.target_transform,
        choices=sorted(_VALID_TRANSFORMS),
        help="Target encoding to use for the 5-bar prediction window (default: %(default)s).",
    )
    args = parser.parse_args()

    config = TrainingConfig(target_transform=args.target_transform)
    logger.info("=== SPY Predictor Preprocessing Pipeline ===")
    logger.info("Raw data directory : %s", DATA_RAW_DIR)
    logger.info("Output directory   : %s", DATA_PROCESSED_DIR)
    logger.info("Target transform   : %s", config.target_transform)

    # Step 1: Load and align raw bar files for every trading day.
    logger.info("Step 1/4 — Loading and aligning raw bar files ...")
    daily_bars = load_aligned_daily_bars(
        raw_data_directory=DATA_RAW_DIR,
        min_bars_required=config.sequence_length,
    )
    logger.info("  %d usable trading days loaded.", len(daily_bars))

    # Step 2: Compute technical indicators for every day.
    logger.info("Step 2/4 — Computing technical indicators ...")
    daily_feature_frames = {
        trading_date: compute_indicators(day_frame, config=config)
        for trading_date, day_frame in daily_bars.items()
    }

    # Step 3: Build stride-1 sliding-window sequences.
    logger.info("Step 3/4 — Building sliding-window sequences ...")
    all_sequences, all_targets, sequence_dates = build_all_sequences(
        daily_feature_frames=daily_feature_frames,
        config=config,
    )
    date_range = f"{min(sequence_dates)} → {max(sequence_dates)}"
    logger.info("  Total sequences : %d", len(all_sequences))
    logger.info("  Sequence shape  : %s", str(all_sequences.shape))
    logger.info("  Targets shape   : %s", str(all_targets.shape))
    logger.info("  Date range      : %s", date_range)

    # Step 4: Chronological train/val/test split and save.
    logger.info("Step 4/4 — Splitting chronologically and saving ...")
    splits = split_sequences_by_day(
        all_sequences=all_sequences,
        all_targets=all_targets,
        sequence_dates=sequence_dates,
        config=config,
    )

    for split_name in ("train", "val", "test"):
        output_path = DATA_PROCESSED_DIR / f"{split_name}.pt"
        SpyDataset.save_split(splits[split_name], output_path)

    logger.info("=== Preprocessing complete. ===")
    logger.info(
        "Train: %d  |  Val: %d  |  Test: %d sequences",
        len(splits["train"]["sequences"]),
        len(splits["val"]["sequences"]),
        len(splits["test"]["sequences"]),
    )


if __name__ == "__main__":
    main()
