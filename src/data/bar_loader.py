"""
Load and align per-day SPY and VIX bar files from the raw data directory.

Each trading day has two source files:
    YYYY-MM-DD_SPY.txt  —  5-second OHLCV bars for SPY
    YYYY-MM-DD_VIX.txt  —  5-second OHLCV bars for VIX

Output: a dict mapping each trading date to a DataFrame of aligned bars,
        indexed by unix_timestamp, with columns:
        spy_open, spy_high, spy_low, spy_close, spy_volume, vix_close

Days with fewer than min_bars_required aligned bars are skipped with a warning.
"""

import re
from datetime import date
from pathlib import Path

import pandas as pd

from src.utils.config import DATA_RAW_DIR, DEFAULT_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)

_FILENAME_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})_(SPY|VIX)\.txt$")

_BAR_FILE_DTYPES = {
    "Date":   "int64",
    "Open":   "float32",
    "High":   "float32",
    "Low":    "float32",
    "Close":  "float32",
    "Volume": "int64",
}


def _parse_bar_file(file_path: Path) -> pd.DataFrame:
    """
    Parse a single bar file into a DataFrame indexed by unix_timestamp.

    Sorts bars by unix_timestamp ascending and logs a warning if any were
    out of order. Drops rows where High < Low (invalid bar) with a warning.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    """
    dataframe = pd.read_csv(file_path, dtype=_BAR_FILE_DTYPES)
    dataframe = dataframe.rename(columns={"Date": "unix_timestamp"})

    if not dataframe["unix_timestamp"].is_monotonic_increasing:
        logger.warning(
            "Out-of-order timestamps in %s — sorting ascending.", file_path.name
        )

    dataframe = dataframe.sort_values("unix_timestamp").reset_index(drop=True)

    invalid_bars = dataframe["High"] < dataframe["Low"]
    if invalid_bars.any():
        logger.warning(
            "%s: dropping %d bars with High < Low.",
            file_path.name,
            int(invalid_bars.sum()),
        )
        dataframe = dataframe[~invalid_bars].reset_index(drop=True)

    return dataframe.set_index("unix_timestamp")


def _align_spy_and_vix(
    spy_bars: pd.DataFrame,
    vix_bars: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join SPY and VIX bars on unix_timestamp.

    Only timestamps present in BOTH files are kept, ensuring every row has
    all required features.

    Returns a DataFrame with columns:
        spy_open, spy_high, spy_low, spy_close, spy_volume, vix_close
    """
    spy_renamed = spy_bars[["Open", "High", "Low", "Close", "Volume"]].rename(
        columns={
            "Open":   "spy_open",
            "High":   "spy_high",
            "Low":    "spy_low",
            "Close":  "spy_close",
            "Volume": "spy_volume",
        }
    )
    vix_close = vix_bars[["Close"]].rename(columns={"Close": "vix_close"})
    return spy_renamed.join(vix_close, how="inner")


def load_aligned_daily_bars(
    raw_data_directory: Path = DATA_RAW_DIR,
    min_bars_required: int = DEFAULT_CONFIG.sequence_length,
) -> dict[date, pd.DataFrame]:
    """
    Discover all SPY+VIX file pairs, align them by timestamp, and return one
    DataFrame per usable trading day.

    Parameters
    ----------
    raw_data_directory : Path
        Directory containing YYYY-MM-DD_SPY.txt and YYYY-MM-DD_VIX.txt files.
    min_bars_required : int
        Minimum number of aligned bars a day must have to be included.
        Days below this threshold are skipped with a logged warning.

    Returns
    -------
    dict[date, pd.DataFrame]
        Chronologically ordered mapping from trading date to aligned bar
        DataFrame. Each DataFrame is indexed by unix_timestamp with columns:
        spy_open, spy_high, spy_low, spy_close, spy_volume, vix_close.

    Raises
    ------
    FileNotFoundError
        If raw_data_directory does not exist (symlink not set up).
    """
    if not raw_data_directory.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_data_directory}. "
            "Ensure the symlink data/raw/daily_data exists."
        )

    spy_files: dict[date, Path] = {}
    vix_files: dict[date, Path] = {}

    for file_path in sorted(raw_data_directory.iterdir()):
        match = _FILENAME_PATTERN.match(file_path.name)
        if match is None:
            continue
        trading_date = date.fromisoformat(match.group(1))
        symbol       = match.group(2)
        if symbol == "SPY":
            spy_files[trading_date] = file_path
        elif symbol == "VIX":
            vix_files[trading_date] = file_path

    common_dates = sorted(set(spy_files) & set(vix_files))
    logger.info("Found %d days with both SPY and VIX files.", len(common_dates))

    aligned_days: dict[date, pd.DataFrame] = {}
    skipped_count = 0

    for trading_date in common_dates:
        spy_bars = _parse_bar_file(spy_files[trading_date])
        vix_bars = _parse_bar_file(vix_files[trading_date])
        aligned  = _align_spy_and_vix(spy_bars, vix_bars)

        if len(aligned) < min_bars_required:
            logger.warning(
                "Skipping %s — %d aligned bars (need ≥ %d).",
                trading_date,
                len(aligned),
                min_bars_required,
            )
            skipped_count += 1
            continue

        aligned_days[trading_date] = aligned

    logger.info(
        "Loaded %d usable trading days (%d skipped).",
        len(aligned_days),
        skipped_count,
    )
    return aligned_days
