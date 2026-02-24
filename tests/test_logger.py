"""
Tests for src/utils/logger.py

Focuses on save_experiment_result() — the append-only experiment log.
All tests redirect EXPERIMENT_LOG_PATH to a tmp_path so the real log is
never touched.
"""

import json
import threading
from pathlib import Path

import pytest

import src.utils.logger as logger_module
from src.utils.logger import save_experiment_result


# ── Helpers ───────────────────────────────────────────────────────────────────

_SAMPLE_CONFIG  = {"d_model": 128, "batch_size": 256}
_SAMPLE_METRICS = {"mae_aggregate": 1.23, "rmse_aggregate": 1.45}


def _save(log_path: Path, name: str = "test_run", notes: str = "") -> None:
    """Write one record to the redirected log path."""
    save_experiment_result(
        experiment_name=name,
        config=_SAMPLE_CONFIG,
        metrics=_SAMPLE_METRICS,
        notes=notes,
    )


# ── Append-only behaviour ─────────────────────────────────────────────────────

class TestAppendOnlyBehaviour:
    def test_creates_log_file_if_it_does_not_exist(self, tmp_path, monkeypatch):
        log_path = tmp_path / "exp.json"
        monkeypatch.setattr(logger_module, "EXPERIMENT_LOG_PATH", log_path)

        assert not log_path.exists()
        _save(log_path)
        assert log_path.exists()

    def test_single_write_produces_one_entry(self, tmp_path, monkeypatch):
        log_path = tmp_path / "exp.json"
        monkeypatch.setattr(logger_module, "EXPERIMENT_LOG_PATH", log_path)

        _save(log_path)

        records = json.loads(log_path.read_text())
        assert len(records) == 1

    def test_two_writes_produce_two_entries(self, tmp_path, monkeypatch):
        log_path = tmp_path / "exp.json"
        monkeypatch.setattr(logger_module, "EXPERIMENT_LOG_PATH", log_path)

        _save(log_path, name="run_a")
        _save(log_path, name="run_b")

        records = json.loads(log_path.read_text())
        assert len(records) == 2

    def test_existing_entries_are_never_modified(self, tmp_path, monkeypatch):
        log_path = tmp_path / "exp.json"
        monkeypatch.setattr(logger_module, "EXPERIMENT_LOG_PATH", log_path)

        _save(log_path, name="first")
        first_snapshot = json.loads(log_path.read_text())[0]

        _save(log_path, name="second")

        records     = json.loads(log_path.read_text())
        first_after = records[0]

        # Every field of the first record must be unchanged.
        assert first_after == first_snapshot

    def test_entries_are_in_chronological_order(self, tmp_path, monkeypatch):
        """Records must appear in write order, newest entry last."""
        log_path = tmp_path / "exp.json"
        monkeypatch.setattr(logger_module, "EXPERIMENT_LOG_PATH", log_path)

        names = ["alpha", "beta", "gamma"]
        for name in names:
            _save(log_path, name=name)

        records = json.loads(log_path.read_text())
        assert [r["experiment_name"] for r in records] == names


# ── Record structure ──────────────────────────────────────────────────────────

class TestRecordStructure:
    def test_record_has_timestamp_field(self, tmp_path, monkeypatch):
        log_path = tmp_path / "exp.json"
        monkeypatch.setattr(logger_module, "EXPERIMENT_LOG_PATH", log_path)

        _save(log_path)

        record = json.loads(log_path.read_text())[0]
        assert "timestamp" in record
        assert isinstance(record["timestamp"], str)
        assert len(record["timestamp"]) > 0

    def test_record_has_experiment_name_field(self, tmp_path, monkeypatch):
        log_path = tmp_path / "exp.json"
        monkeypatch.setattr(logger_module, "EXPERIMENT_LOG_PATH", log_path)

        _save(log_path, name="my_run")

        record = json.loads(log_path.read_text())[0]
        assert record["experiment_name"] == "my_run"

    def test_record_has_config_field(self, tmp_path, monkeypatch):
        log_path = tmp_path / "exp.json"
        monkeypatch.setattr(logger_module, "EXPERIMENT_LOG_PATH", log_path)

        _save(log_path)

        record = json.loads(log_path.read_text())[0]
        assert "config"  in record
        assert record["config"] == _SAMPLE_CONFIG

    def test_record_has_metrics_field(self, tmp_path, monkeypatch):
        log_path = tmp_path / "exp.json"
        monkeypatch.setattr(logger_module, "EXPERIMENT_LOG_PATH", log_path)

        _save(log_path)

        record = json.loads(log_path.read_text())[0]
        assert "metrics"  in record
        assert record["metrics"] == _SAMPLE_METRICS

    def test_notes_field_is_preserved(self, tmp_path, monkeypatch):
        log_path = tmp_path / "exp.json"
        monkeypatch.setattr(logger_module, "EXPERIMENT_LOG_PATH", log_path)

        _save(log_path, notes="best_epoch=42")

        record = json.loads(log_path.read_text())[0]
        assert record["notes"] == "best_epoch=42"


# ── Concurrent writes ─────────────────────────────────────────────────────────

class TestConcurrentWrites:
    def test_concurrent_writes_all_appear_in_log(self, tmp_path, monkeypatch):
        """
        N threads each writing one record must collectively produce N records.

        This is a best-effort check: the current logger uses file-level
        read-write which is NOT atomic under true concurrency.  The test
        verifies that *serial* interleaving (the common case on a single core)
        preserves all entries.  If the implementation is later upgraded to use
        file locking, this test provides a regression baseline.
        """
        log_path   = tmp_path / "exp.json"
        monkeypatch.setattr(logger_module, "EXPERIMENT_LOG_PATH", log_path)

        n_threads  = 8
        barrier    = threading.Barrier(n_threads)
        errors     = []

        def worker(index: int) -> None:
            try:
                barrier.wait()   # all threads start at the same time
                save_experiment_result(
                    experiment_name=f"thread_{index}",
                    config=_SAMPLE_CONFIG,
                    metrics=_SAMPLE_METRICS,
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Threads raised exceptions: {errors}"
        records = json.loads(log_path.read_text())
        # The in-process lock serialises all writes, so every thread's record
        # must appear exactly once.
        assert len(records) == n_threads
