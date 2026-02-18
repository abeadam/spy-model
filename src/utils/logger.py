"""
Structured logging and experiment result tracking.
Every training run and evaluation appends a record to results/experiment_log.json.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.utils.config import EXPERIMENT_LOG_PATH


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes formatted output to stdout."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


def save_experiment_result(
    experiment_name: str,
    config: dict,
    metrics: dict,
    notes: str = "",
) -> None:
    """
    Append a structured result record to the experiment log.
    Old results are never overwritten — each run adds a new entry.
    """
    result_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "experiment_name": experiment_name,
        "config": config,
        "metrics": metrics,
        "notes": notes,
    }

    existing_records: list[dict] = []
    if EXPERIMENT_LOG_PATH.exists():
        with open(EXPERIMENT_LOG_PATH) as log_file:
            existing_records = json.load(log_file)

    existing_records.append(result_record)

    with open(EXPERIMENT_LOG_PATH, "w") as log_file:
        json.dump(existing_records, log_file, indent=2)

    logger = get_logger(__name__)
    logger.info(f"Experiment result saved → {EXPERIMENT_LOG_PATH}")
    logger.info(f"Metrics: {metrics}")
