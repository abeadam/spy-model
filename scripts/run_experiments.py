#!/usr/bin/env python
"""
Sequential experiment runner: preprocess → train for each target transform.

Runs the four scaling variants in order:
    1. basis_points        — percent change × 10,000
    2. signed_log          — sign(x) × log1p(|x|)
    3. vol_normalized      — percent change ÷ local realized vol
    4. vol_normalized_log  — log return ÷ local realized vol (log space)

Results are appended to results/experiment_log.json after each run.
The experiment_name field is "training_run" and the config snapshot records
the target_transform used, so all four results are directly comparable.

Usage:
    /Users/abeadam/dev/model/Stock-Model/.venv/bin/python scripts/run_experiments.py
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import EXPERIMENT_LOG_PATH
from src.utils.logger import get_logger

logger = get_logger("run_experiments")

PYTHON    = sys.executable
PREPROCESS = str(Path(__file__).resolve().parent / "preprocess.py")
TRAIN      = str(Path(__file__).resolve().parent / "train.py")

TRANSFORMS = [
    "basis_points",           # #2: percent change × 10,000
    "signed_log",             # #3: sign(x) × log1p(|x|)
    "vol_normalized",         # #4: percent change ÷ local realized vol
    "vol_normalized_log",     # #4b: log return ÷ local vol (dimensionless Sharpe)
    "vol_normalized_log_bp",  # #5: (10,000 × log return) ÷ local vol (bp-Sharpe)
]


def _count_existing_records() -> int:
    if not EXPERIMENT_LOG_PATH.exists():
        return 0
    with open(EXPERIMENT_LOG_PATH) as f:
        return len(json.load(f))


def _get_last_record() -> dict:
    with open(EXPERIMENT_LOG_PATH) as f:
        return json.load(f)[-1]


def _run(cmd: list[str], label: str) -> None:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("%s failed with exit code %d — aborting.", label, result.returncode)
        sys.exit(result.returncode)


def _format_metrics(metrics: dict, prediction_steps: int = 5) -> str:
    lines = []
    for step in range(1, prediction_steps + 1):
        lines.append(
            f"  Step {step}: MAE={metrics[f'mae_step_{step}']:.6f}  "
            f"RMSE={metrics[f'rmse_step_{step}']:.6f}  "
            f"R²={metrics[f'r2_step_{step}']:.4f}  "
            f"DirAcc={metrics[f'directional_accuracy_step_{step}']:.2f}%"
        )
    lines.append(
        f"  Aggr : MAE={metrics['mae_aggregate']:.6f}  "
        f"RMSE={metrics['rmse_aggregate']:.6f}  "
        f"R²={metrics['r2_aggregate']:.4f}  "
        f"DirAcc={metrics['directional_accuracy_aggregate']:.2f}%"
    )
    return "\n".join(lines)


def main() -> None:
    logger.info("=== SPY Predictor Experiment Suite ===")
    logger.info("Transforms to run: %s", TRANSFORMS)

    results: dict[str, dict] = {}

    for transform in TRANSFORMS:
        logger.info("")
        logger.info("══════════════════════════════════════════")
        logger.info("  Transform: %s", transform)
        logger.info("══════════════════════════════════════════")

        before = _count_existing_records()

        _run([PYTHON, PREPROCESS, "--target-transform", transform], f"preprocess({transform})")
        _run([PYTHON, TRAIN,      "--target-transform", transform], f"train({transform})")

        after = _count_existing_records()
        if after <= before:
            logger.error("No new log entry for transform=%s — something went wrong.", transform)
            sys.exit(1)

        record  = _get_last_record()
        metrics = record["metrics"]
        results[transform] = metrics

        logger.info("Results for %s:", transform)
        logger.info(_format_metrics(metrics))

    # ── Final comparison table ────────────────────────────────────────────────
    logger.info("")
    logger.info("══════════════════════════════════════════")
    logger.info("  FINAL COMPARISON")
    logger.info("══════════════════════════════════════════")
    logger.info("%-22s  %10s  %10s  %8s  %8s", "Transform", "MAE_aggr", "RMSE_aggr", "R²_aggr", "DirAcc%")
    logger.info("%-22s  %10s  %10s  %8s  %8s", "-" * 22, "-" * 10, "-" * 10, "-" * 8, "-" * 8)
    for transform, metrics in results.items():
        logger.info(
            "%-22s  %10.6f  %10.6f  %8.4f  %8.2f%%",
            transform,
            metrics["mae_aggregate"],
            metrics["rmse_aggregate"],
            metrics["r2_aggregate"],
            metrics["directional_accuracy_aggregate"],
        )
    logger.info("=== Experiment suite complete. ===")


if __name__ == "__main__":
    main()
