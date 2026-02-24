"""
Tests for src/data/bar_loader.py

Uses temporary directories and synthetic CSV data — does not touch real bar files.
"""

import csv
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.bar_loader import (
    _align_spy_and_vix,
    _parse_bar_file,
    load_aligned_daily_bars,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_bar_csv(
    file_path: Path,
    rows: list[dict],
) -> None:
    """Write a minimal bar CSV file to file_path."""
    with open(file_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["Date", "Open", "High", "Low", "Close", "Volume"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _make_bar_rows(
    count: int,
    start_timestamp: int = 1_700_000_000,
    price: float = 500.0,
    volume: int = 1_000,
) -> list[dict]:
    """Generate `count` synthetic 5-second bar rows at a fixed price."""
    return [
        {
            "Date":   start_timestamp + i * 5,
            "Open":   price,
            "High":   price + 0.10,
            "Low":    price - 0.10,
            "Close":  price,
            "Volume": volume,
        }
        for i in range(count)
    ]


def _write_day_pair(
    directory: Path,
    trading_date: str,
    spy_count: int,
    vix_count: int,
    spy_price: float = 500.0,
    vix_price: float = 18.0,
    start_timestamp: int = 1_700_000_000,
) -> None:
    """Write one SPY + one VIX bar file for a given date string."""
    _write_bar_csv(
        directory / f"{trading_date}_SPY.txt",
        _make_bar_rows(spy_count, start_timestamp, price=spy_price),
    )
    _write_bar_csv(
        directory / f"{trading_date}_VIX.txt",
        _make_bar_rows(vix_count, start_timestamp, price=vix_price, volume=0),
    )


# ── _parse_bar_file ───────────────────────────────────────────────────────────

class TestParseBarFile:
    def test_parses_columns_correctly(self, tmp_path):
        file_path = tmp_path / "2024-01-02_SPY.txt"
        _write_bar_csv(file_path, _make_bar_rows(10))
        result = _parse_bar_file(file_path)

        expected_columns = {"Open", "High", "Low", "Close", "Volume"}
        assert set(result.columns) == expected_columns
        assert result.index.name == "unix_timestamp"

    def test_index_is_sorted_ascending(self, tmp_path):
        file_path = tmp_path / "test.txt"
        # Write rows in reverse order.
        rows = _make_bar_rows(10, start_timestamp=1_700_000_000)
        rows_reversed = list(reversed(rows))
        _write_bar_csv(file_path, rows_reversed)

        result = _parse_bar_file(file_path)
        assert result.index.is_monotonic_increasing

    def test_drops_invalid_bars_where_high_less_than_low(self, tmp_path):
        file_path = tmp_path / "test.txt"
        rows = _make_bar_rows(5)
        # Corrupt bar 2: High < Low.
        rows[2]["High"] = 499.0
        rows[2]["Low"]  = 500.0
        _write_bar_csv(file_path, rows)

        result = _parse_bar_file(file_path)
        assert len(result) == 4  # One bar dropped.

    def test_row_count_matches_input(self, tmp_path):
        file_path = tmp_path / "test.txt"
        _write_bar_csv(file_path, _make_bar_rows(200))
        result = _parse_bar_file(file_path)
        assert len(result) == 200


# ── _align_spy_and_vix ────────────────────────────────────────────────────────

class TestAlignSpyAndVix:
    def _make_indexed_df(self, timestamps, price=500.0, volume=1000):
        """Build a minimal DataFrame indexed by unix_timestamp."""
        return pd.DataFrame(
            {
                "Open":   price,
                "High":   price + 0.1,
                "Low":    price - 0.1,
                "Close":  price,
                "Volume": volume,
            },
            index=pd.Index(timestamps, name="unix_timestamp"),
        )

    def test_inner_join_keeps_only_common_timestamps(self):
        spy_ts = [100, 105, 110, 115]
        vix_ts = [100, 110, 120]          # 105 and 115 missing; 120 extra
        spy_df = self._make_indexed_df(spy_ts)
        vix_df = self._make_indexed_df(vix_ts, price=18.0, volume=0)

        result = _align_spy_and_vix(spy_df, vix_df)
        assert list(result.index) == [100, 110]

    def test_output_has_correct_column_names(self):
        timestamps = [100, 105, 110]
        spy_df = self._make_indexed_df(timestamps)
        vix_df = self._make_indexed_df(timestamps, price=18.0, volume=0)

        result = _align_spy_and_vix(spy_df, vix_df)
        expected_columns = {"spy_open", "spy_high", "spy_low", "spy_close", "spy_volume", "vix_close"}
        assert set(result.columns) == expected_columns

    def test_vix_close_values_are_correct(self):
        timestamps = [100, 105]
        spy_df = self._make_indexed_df(timestamps, price=500.0)
        vix_df = self._make_indexed_df(timestamps, price=18.5, volume=0)

        result = _align_spy_and_vix(spy_df, vix_df)
        assert np.allclose(result["vix_close"].values, 18.5)


# ── load_aligned_daily_bars ───────────────────────────────────────────────────

class TestLoadAlignedDailyBars:
    def test_raises_if_directory_does_not_exist(self, tmp_path):
        missing_dir = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="symlink"):
            load_aligned_daily_bars(raw_data_directory=missing_dir)

    def test_discovers_correct_number_of_days(self, tmp_path):
        _write_day_pair(tmp_path, "2024-01-02", spy_count=100, vix_count=100)
        _write_day_pair(tmp_path, "2024-01-03", spy_count=100, vix_count=100)
        result = load_aligned_daily_bars(raw_data_directory=tmp_path, min_bars_required=60)
        assert len(result) == 2

    def test_ignores_files_with_unexpected_names(self, tmp_path):
        _write_day_pair(tmp_path, "2024-01-02", spy_count=100, vix_count=100)
        # Extra file that should be ignored.
        (tmp_path / "README.txt").write_text("ignored")
        (tmp_path / "2024-01-03_SPX.txt").write_text("ignored")
        result = load_aligned_daily_bars(raw_data_directory=tmp_path, min_bars_required=60)
        assert len(result) == 1

    def test_skips_days_with_fewer_than_min_bars(self, tmp_path):
        _write_day_pair(tmp_path, "2024-01-02", spy_count=100, vix_count=100)
        _write_day_pair(tmp_path, "2024-01-03", spy_count=10,  vix_count=10)  # Too short.
        result = load_aligned_daily_bars(raw_data_directory=tmp_path, min_bars_required=60)
        assert len(result) == 1
        assert date(2024, 1, 2) in result
        assert date(2024, 1, 3) not in result

    def test_result_is_ordered_chronologically(self, tmp_path):
        _write_day_pair(tmp_path, "2024-01-04", spy_count=100, vix_count=100)
        _write_day_pair(tmp_path, "2024-01-02", spy_count=100, vix_count=100)
        _write_day_pair(tmp_path, "2024-01-03", spy_count=100, vix_count=100)
        result = load_aligned_daily_bars(raw_data_directory=tmp_path, min_bars_required=60)
        assert list(result.keys()) == sorted(result.keys())

    def test_result_has_correct_column_names(self, tmp_path):
        _write_day_pair(tmp_path, "2024-01-02", spy_count=100, vix_count=100)
        result = load_aligned_daily_bars(raw_data_directory=tmp_path, min_bars_required=60)
        frame = result[date(2024, 1, 2)]
        expected = {"spy_open", "spy_high", "spy_low", "spy_close", "spy_volume", "vix_close"}
        assert set(frame.columns) == expected

    def test_output_is_deterministic(self, tmp_path):
        _write_day_pair(tmp_path, "2024-01-02", spy_count=100, vix_count=100)
        result_a = load_aligned_daily_bars(raw_data_directory=tmp_path, min_bars_required=60)
        result_b = load_aligned_daily_bars(raw_data_directory=tmp_path, min_bars_required=60)
        pd.testing.assert_frame_equal(
            result_a[date(2024, 1, 2)],
            result_b[date(2024, 1, 2)],
        )
