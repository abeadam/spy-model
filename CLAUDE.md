# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Python interpreter**: `/Users/abeadam/dev/model/Stock-Model/.venv/bin/python`
- **pip**: `/Users/abeadam/dev/model/Stock-Model/.venv/bin/pip`
- Always prefix commands with the full venv path — do NOT activate the venv.

## Commands

```bash
# Run training
/Users/abeadam/dev/model/Stock-Model/.venv/bin/python scripts/train.py

# Run evaluation
/Users/abeadam/dev/model/Stock-Model/.venv/bin/python scripts/evaluate.py

# Run all tests
/Users/abeadam/dev/model/Stock-Model/.venv/bin/python -m pytest tests/ -v

# Run a single test file
/Users/abeadam/dev/model/Stock-Model/.venv/bin/python -m pytest tests/test_model.py -v

# Run a single test by name
/Users/abeadam/dev/model/Stock-Model/.venv/bin/python -m pytest tests/test_model.py::test_function_name -v
```

## Project Structure

```
spy-model/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── model/          # Model architecture and training logic
│   ├── evaluation/     # Metrics, benchmarks, and result tracking
│   └── utils/          # Shared config, logging, helpers
├── scripts/
│   ├── train.py        # Entry point for training
│   └── evaluate.py     # Entry point for standalone evaluation
├── tests/              # Pytest tests — run after every model change
├── results/            # JSON/CSV experiment logs (never delete old results)
└── data/
    ├── raw/            # Original input data, read-only
    └── processed/      # Preprocessed/cached data
```

## Architecture Principles

- **Config-driven.** All hyperparameters and paths live in `src/utils/config.py` — never hardcoded in training or model files.
- **Evaluation is fast.** The inference path (forward pass + metrics) must be optimized. Training can be slow; evaluation cannot.
- **Every task ends with a test.** After any model or data change, run the test suite. A task is not done until tests pass.
- **Results are always saved.** Every training run and evaluation appends a structured record to `results/experiment_log.json` including timestamp, config snapshot, and all metrics.

## Active Technologies
- Python 3.13 (`/Users/abeadam/dev/model/Stock-Model/.venv/bin/python`) + PyTorch 2.x, numpy, pandas, scikit-learn, pytest (all in shared venv) (001-spy-predictor)
- Files — raw `.txt` bars, processed `.pt` tensors, JSON experiment log, `.pth` checkpoints (001-spy-predictor)

## Key Implementation Details

### Data
- Raw data is at `data/raw/daily_data/` (symlink — never auto-create this path).
- `scripts/preprocess.py` builds `data/processed/train.pt`, `val.pt`, `test.pt` (1.5M+ sequences).
- Per-feature z-score normalization is computed per-window (not globally), handled in `src/data/normalizer.py`.
- The 9 features are ordered: `spy_close` (index 0), `vix_close`, `ema_5`, `ema_9`, `ema_12`, `rsi_7`, `vwap_deviation`, `price_momentum_5`, `volume_momentum_5`.

### Model (`src/model/`)
- Encoder-only transformer: d_model=128, n_heads=4, n_layers=3, ffn_dim=512 ≈ 595K params.
- `SpyPredictor.predict()` accepts **raw un-normalized** windows (shape 60×9); normalizes internally.
- Targets are **percent changes** from bar 60's close: `(close[t+k] - close[t]) / close[t]` for k=1..5. No denormalization needed.
- Directional signal: `"up" if predicted_changes[-1] > 0.0 else "down"` (not compared to last_known_close).
- Checkpoints (`models/spy_predictor_best.pth`) store state_dict + config dict + metadata (best_epoch, best_val_loss, training_history).

### Training
- `scripts/train.py --resume <ckpt>` supports resuming. `torch.compile()` is enabled by default.
- Early stopping on val loss with patience=30. LR scheduler: CosineAnnealingWarmRestarts(T_0=50).
- `src/utils/logger.py` uses a threading.Lock for safe concurrent writes to the experiment log.

### Evaluation
- `scripts/evaluate.py --checkpoint <ckpt> --data <test.pt>` runs standalone evaluation.
- Latency is sampled from `_N_LATENCY_SAMPLES=50` synthetic windows; `torch.compile()` used in main().
- `run_evaluation()` sets `use_compile=False` by default (keeps tests fast).

### Test suite
- 140 tests across 10 files; runs in ~1.3 seconds total.
- Monkeypatch experiment log path in any test that calls `save_experiment_result()` via `monkeypatch.setattr("src.utils.logger.EXPERIMENT_LOG_PATH", tmp_path / "log.json")`.
- Early stopping tests use monkeypatch on `_run_epoch` to inject controlled val loss sequences.

## Recent Changes
- Phases 4–7 complete: trainer, evaluate script, logger (with thread lock), normalizer tests, torch.compile support.
- **Target representation changed to percent change**: targets are now `(close[t+k] - close[t]) / close[t]` — processed .pt files must be regenerated with `scripts/preprocess.py` before training.
