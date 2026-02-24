#!/usr/bin/env python
"""
Experiment set 2: Extended indicators (26 features), 120-bar lookback, Huber loss.

Runs 3 target-transform experiments sequentially, reusing the same bar-loading
and indicator-computation pass for all transforms.

Transforms:
    percent_change          — simple percent return (Huber delta = 0.005)
    vol_normalized_log      — log return / local window vol (Huber delta = 3.0)
    vol_normalized_log_prevday — log return / prior-day vol (Huber delta = 3.0)

Each transform:
    1. Builds sequences from the shared indicator frames.
    2. Saves train/val/test splits to data/processed/<transform>/.
    3. Launches train.py as a subprocess.

At the end, prints a comparison table of best val loss and test metrics.

Usage:
    /Users/abeadam/dev/model/Stock-Model/.venv/bin/python scripts/run_experiment_set2.py
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bar_loader import load_aligned_daily_bars
from src.data.dataset import SpyDataset, split_sequences_by_day
from src.data.indicator_computer import compute_indicators
from src.data.sequence_builder import build_all_sequences
from src.utils.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DEFAULT_CONFIG,
    EXPERIMENT_LOG_PATH,
    TrainingConfig,
)
from src.utils.logger import get_logger

logger = get_logger("run_experiment_set2")

PYTHON = "/Users/abeadam/dev/model/Stock-Model/.venv/bin/python"
TRAIN  = str(Path(__file__).resolve().parent / "train.py")

# (transform, huber_delta) pairs — delta chosen to match scale of the target distribution.
# percent_change targets are ~[-0.01, 0.01]; delta=0.005 clips outliers beyond ±0.5%.
# vol_normalized_log targets are ~[-5, +5] (Sharpe units); delta=3.0 clips beyond ±3σ.
EXPERIMENTS: list[tuple[str, float]] = [
    ("percent_change",              0.005),
    ("vol_normalized_log",          3.0),
    ("vol_normalized_log_prevday",  3.0),
]


def _print_section(title: str) -> None:
    separator = "=" * 72
    print(f"\n{separator}")
    print(f"  {title}")
    print(separator)


def _load_most_recent_log_entry() -> dict | None:
    """Return the most recently appended entry in the experiment log, or None."""
    if not EXPERIMENT_LOG_PATH.exists():
        return None
    with open(EXPERIMENT_LOG_PATH) as f:
        records = json.load(f)
    return records[-1] if records else None


def main() -> None:
    start_time = datetime.now()
    _print_section("Experiment Set 2: 26-feature, 120-bar, Huber loss")

    # ── Step 1: Load and align raw bars (once for all experiments) ────────────
    _print_section("Step 1/3 — Loading raw bar files")
    daily_bars = load_aligned_daily_bars(
        raw_data_directory=DATA_RAW_DIR,
        min_bars_required=DEFAULT_CONFIG.sequence_length,
    )
    print(f"  {len(daily_bars)} usable trading days loaded.")

    # ── Step 2: Compute indicators (once for all experiments) ─────────────────
    _print_section("Step 2/3 — Computing indicators (26 features)")
    daily_feature_frames = {
        trading_date: compute_indicators(day_frame, config=DEFAULT_CONFIG)
        for trading_date, day_frame in daily_bars.items()
    }
    print(f"  Indicators computed for {len(daily_feature_frames)} days.")

    # ── Step 3: Run each experiment ───────────────────────────────────────────
    _print_section("Step 3/3 — Running experiments")

    results: list[dict] = []

    for transform, huber_delta in EXPERIMENTS:
        print(f"\n{'─' * 72}")
        print(f"  Transform : {transform}")
        print(f"  Huber δ   : {huber_delta}")
        print(f"{'─' * 72}")

        config = TrainingConfig(
            target_transform=transform,
            huber_loss_delta=huber_delta,
        )

        # Build sequences for this transform.
        print("  Building sequences ...")
        all_sequences, all_targets, sequence_dates = build_all_sequences(
            daily_feature_frames=daily_feature_frames,
            config=config,
        )
        print(f"  Total sequences: {len(all_sequences):,}")

        # Split chronologically.
        splits = split_sequences_by_day(
            all_sequences=all_sequences,
            all_targets=all_targets,
            sequence_dates=sequence_dates,
            config=config,
        )
        print(
            f"  Split sizes — train: {len(splits['train']['sequences']):,}  "
            f"val: {len(splits['val']['sequences']):,}  "
            f"test: {len(splits['test']['sequences']):,}"
        )

        # Save splits to a transform-specific subdirectory.
        out_dir = DATA_PROCESSED_DIR / transform
        out_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "val", "test"):
            SpyDataset.save_split(splits[split_name], out_dir / f"{split_name}.pt")
        print(f"  Splits saved to {out_dir}")

        # Launch training subprocess.
        print("  Launching training ...")
        cmd = [
            PYTHON, TRAIN,
            "--target-transform", transform,
            "--data-dir",         str(out_dir),
            "--huber-delta",      str(huber_delta),
        ]
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            print(f"  [WARNING] Training subprocess exited with code {result.returncode}.")
            results.append({"transform": transform, "huber_delta": huber_delta, "status": "FAILED"})
            continue

        # Pull the most-recent log entry written by train.py.
        log_entry = _load_most_recent_log_entry()
        if log_entry and log_entry.get("experiment_name") == "training_run":
            metrics = log_entry.get("metrics", {})
            results.append({
                "transform":   transform,
                "huber_delta": huber_delta,
                "status":      "OK",
                "mae_agg":     metrics.get("mae_aggregate",                  float("nan")),
                "dir_acc":     metrics.get("directional_accuracy_aggregate", float("nan")),
                "r2_agg":      metrics.get("r2_aggregate",                   float("nan")),
            })
        else:
            results.append({"transform": transform, "huber_delta": huber_delta, "status": "LOG_MISSING"})

    # ── Comparison table ──────────────────────────────────────────────────────
    elapsed = datetime.now() - start_time
    _print_section(f"Results (elapsed: {elapsed})")

    header = f"{'Transform':<32}  {'δ':>6}  {'MAE_agg':>9}  {'DirAcc%':>9}  {'R²_agg':>9}  {'Status'}"
    print(header)
    print("─" * len(header))
    for r in results:
        if r["status"] == "OK":
            print(
                f"{r['transform']:<32}  {r['huber_delta']:>6.3f}  "
                f"{r['mae_agg']:>9.5f}  {r['dir_acc']:>9.1f}  "
                f"{r['r2_agg']:>9.4f}  {r['status']}"
            )
        else:
            print(f"{r['transform']:<32}  {r['huber_delta']:>6.3f}  {'—':>9}  {'—':>9}  {'—':>9}  {r['status']}")

    print()


if __name__ == "__main__":
    main()
