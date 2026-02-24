#!/usr/bin/env python
"""
Run experiment 5: vol_normalized_log_bp.

This is the true "combined" variant — log return expressed in basis points,
normalized by local realized volatility in raw log-return units:

    target[k] = 10_000 × log(close[t+k] / close[t]) / std(window_log_returns)

Targets land in the range ≈ [-50, +50] (bp-Sharpe units).

Run AFTER scripts/run_experiments.py completes (experiments 1–4).

Usage:
    /Users/abeadam/dev/model/Stock-Model/.venv/bin/python scripts/run_experiment5.py
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import EXPERIMENT_LOG_PATH
from src.utils.logger import get_logger

logger   = get_logger("run_experiment5")
PYTHON   = sys.executable
SCRIPTS  = Path(__file__).resolve().parent
TRANSFORM = "vol_normalized_log_bp"


def _run(cmd: list[str], label: str) -> None:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("%s failed (exit %d).", label, result.returncode)
        sys.exit(result.returncode)


def main() -> None:
    logger.info("=== Experiment 5: %s ===", TRANSFORM)

    _run([PYTHON, str(SCRIPTS / "preprocess.py"), "--target-transform", TRANSFORM],
         f"preprocess({TRANSFORM})")
    _run([PYTHON, str(SCRIPTS / "train.py"),      "--target-transform", TRANSFORM],
         f"train({TRANSFORM})")

    with open(EXPERIMENT_LOG_PATH) as f:
        records = json.load(f)
    last    = records[-1]
    metrics = last["metrics"]

    logger.info("Results for %s:", TRANSFORM)
    for step in range(1, 6):
        logger.info(
            "  Step %d: MAE=%.6f  RMSE=%.6f  R²=%.4f  DirAcc=%.2f%%",
            step,
            metrics[f"mae_step_{step}"],
            metrics[f"rmse_step_{step}"],
            metrics[f"r2_step_{step}"],
            metrics[f"directional_accuracy_step_{step}"],
        )
    logger.info(
        "  Aggr : MAE=%.6f  RMSE=%.6f  R²=%.4f  DirAcc=%.2f%%",
        metrics["mae_aggregate"],
        metrics["rmse_aggregate"],
        metrics["r2_aggregate"],
        metrics["directional_accuracy_aggregate"],
    )
    logger.info("=== Experiment 5 complete. ===")


if __name__ == "__main__":
    main()
