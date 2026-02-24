# Implementation Plan: SPY Intraday 5-Step Price Predictor

**Branch**: `001-spy-predictor` | **Date**: 2026-02-18 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/001-spy-predictor/spec.md`

---

## Summary

Build a PyTorch encoder-only transformer that takes a 60-bar window of SPY 5-second OHLCV bars + VIX close bars + derived technical indicators and predicts the next 5 SPY closing-price percent changes relative to bar 60's close. Inference must complete in under 100ms (latency logged and flagged on violation). Training is unconstrained in time and targets the best validation loss checkpoint. All results are appended to a persistent experiment log.

---

## Technical Context

**Language/Version**: Python 3.13 (`/Users/abeadam/dev/model/Stock-Model/.venv/bin/python`)
**C/C++ Extensions**: pybind11 + Clang for indicator computation hot path; NumPy fallback if extension not built
**Primary Dependencies**: PyTorch 2.x, numpy, pandas, scikit-learn, pytest (all in shared venv)
**Storage**: Files — raw `.txt` bars, processed `.pt` tensors, JSON experiment log, `.pth` checkpoints
**Testing**: pytest — full suite must pass in under 60 seconds
**Target Platform**: macOS with Apple MPS (Metal) as the primary compute device. MPS MUST be used when available; CPU is the fallback only when MPS is unavailable.
**Performance Goal**: Single-sample inference (pre-loaded model) ≤ 100ms end-to-end, logged every call
**Constraints**: Sequence windows must not span day boundaries; chronological train/val/test split only
**Scale/Scope**: 304 trading days × ~4,680 bars/day ≈ 1.4M bar records; ~1.37M training sequences

---

## Constitution Check

| Principle | Gate | Status |
|---|---|---|
| I. Inference-First | All layers on inference path must be benchmarked. `torch.compile()` enabled by default. ONNX export path documented. | ✅ |
| II. Train-to-Best | No epoch time limit. Early stopping on validation loss only. Best checkpoint always saved. | ✅ |
| III. Test-and-Track | Test suite covers data, indicators, metrics, model I/O. Every run appends to experiment log. | ✅ |
| IV. Staff-Level Quality | All modules single-responsibility. No magic numbers. Variable names fully descriptive. Post-task review required. | ✅ |
| V. Config-Driven | All hyperparameters and paths in `src/utils/config.py`. No constants in model or training files. | ✅ |

---

## Project Structure

### Documentation (this feature)

```
specs/001-spy-predictor/
├── spec.md
├── plan.md              ← this file
├── research.md
├── data-model.md
├── tasks.md
└── checklists/
    └── requirements.md
```

### Source Code

```
spy-model/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── bar_loader.py         # Load + merge per-day SPY + VIX .txt files
│   │   ├── indicator_computer.py # Compute EMA, RSI, VWAP deviation, momentum
│   │   ├── normalizer.py         # Per-feature z-score normalization (window-based)
│   │   ├── sequence_builder.py   # Slice 60-bar input windows + 5-bar target windows
│   │   └── dataset.py            # PyTorch Dataset wrapping processed sequences
│   ├── model/
│   │   ├── __init__.py
│   │   ├── attention.py          # MultiHeadSelfAttention using scaled_dot_product_attention
│   │   ├── encoder_block.py      # Single transformer encoder block (attention + FFN + norms)
│   │   ├── spy_predictor.py      # Full model: embedding → N encoder blocks → projection head
│   │   └── checkpoint.py         # Save / load best checkpoint with metadata
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop with early stopping on val loss
│   │   └── scheduler.py          # Learning rate scheduling (cosine annealing with warmup)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # MAE, RMSE, R², directional accuracy (per-step + aggregate)
│   │   └── latency_tracker.py    # Measure + log inference latency; flag violations > 100ms
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # All hyperparameters + paths (single source of truth)
│       └── logger.py             # Structured stdout logging + experiment_log.json appender
├── scripts/
│   ├── preprocess.py             # One-time: load all raw bars → save processed tensors
│   ├── train.py                  # Entry point: load processed data → train → log results
│   └── evaluate.py               # Entry point: load model + test set → metrics → log results
├── tests/
│   ├── __init__.py
│   ├── test_bar_loader.py
│   ├── test_indicator_computer.py
│   ├── test_normalizer.py
│   ├── test_sequence_builder.py
│   ├── test_metrics.py           # Already written ✓
│   ├── test_latency_tracker.py
│   └── test_spy_predictor.py     # Input shape → output shape; determinism; no NaN outputs
├── data/
│   ├── raw/                      # Symlink → /Users/abeadam/dev/interactive-broker-python/Updated Stats/daily_data (read-only, never copied)
│   └── processed/                # Cached .pt tensors output by preprocess.py
├── models/                       # Saved .pth checkpoints
└── results/
    └── experiment_log.json       # Append-only experiment history
```

---

## Phase 0: Research (Complete)

See `research.md`. All architecture decisions resolved:

- **Architecture**: Encoder-only transformer, direct 5-output projection head
- **Attention**: `torch.scaled_dot_product_attention` (full attention, 60 tokens)
- **Speed**: MPS (Metal) as primary device; `torch.compile()` for graph optimization; ONNX export optional for non-MPS deployments
- **Size**: d_model=128, n_heads=4, n_layers=3, ffn_dim=512 (~105K params)
- **C/C++**: pybind11 + Clang C++ extension as primary fast-path for indicator computation; NumPy fallback if extension not built
- **Indicators**: EMA(5), EMA(9), EMA(12), RSI(7), VWAP deviation (60-bar), price momentum (5-bar), volume momentum (5-bar)
- **Normalization**: Per-feature z-score over each 60-bar window at inference time
- **Split**: Chronological — train 80% / val 10% / test 10% by trading day

---

## Phase 1: Design

### Data Model

See `data-model.md` for full entity definitions.

**Key data shapes**:

| Entity | Shape | dtype | Notes |
|---|---|---|---|
| Raw bar file row | — | CSV | timestamp (Unix epoch), O, H, L, C, V |
| Aligned daily frame | (N_bars, 2) | float32 | SPY close + VIX close per bar, same-day only |
| Feature vector | (N_features,) | float32 | 9 features: close, VIX, EMA(5,9,12), RSI(7), VWAP dev, price mom, vol mom |
| Input sequence | (60, N_features) | float32 | 60-bar window, normalized per-feature |
| Target sequence | (5,) | float32 | Next 5 SPY close percent changes from bar 60's close |
| Model output | (batch, 5) | float32 | 5 predicted SPY close percent changes |

### Model Architecture

```
Input: (batch, 60, N_features)   # N_features = 9
  ↓
Linear projection → (batch, 60, d_model=128)
  ↓
+ Sinusoidal positional encoding
  ↓
[EncoderBlock × 3]
  each block:
    MultiHeadSelfAttention(d_model=128, n_heads=4)   # scaled_dot_product_attention
    → LayerNorm → residual
    FFN(128 → 512 → 128, GELU activation)
    → LayerNorm → residual
  ↓
Select last time-step output → (batch, 128)
  ↓
Linear(128 → 5) → (batch, 5)      # 5 predicted SPY close percent changes
```

**Why last time-step**: The model sees all 60 bars via bidirectional attention; the last position aggregates the full context. Simpler and faster than pooling across all 60 positions.

**Positional encoding**: Sinusoidal (fixed, no learned parameters) — faster and sufficient for 60-position sequences.

### Inference Latency Budget

| Step | Estimated time (MPS) |
|---|---|
| Indicator computation (60 bars, pre-loaded) | ~3ms |
| Per-window normalization | <1ms |
| Tensor construction + MPS device transfer | ~2ms |
| Model forward pass (3 encoder blocks, 60 tokens, MPS) | ~5-15ms |
| Direction computation (sign of percent change) | <1ms |
| Latency logging | <1ms |
| **Total** | **~12-22ms** (well under 100ms) |

**Device selection**: `torch.device("mps")` when `torch.backends.mps.is_available()`, else `torch.device("cpu")`. Device is resolved once at startup in `src/utils/config.py` and shared across all modules.

### Indicator Specification

All computed from the 60-bar input window plus a larger history buffer (for warmup):

| Indicator | Period | Formula | Warmup needed |
|---|---|---|---|
| EMA(5) | 5 bars | Exponential weighted mean, α=2/(5+1) | 15 bars |
| EMA(9) | 9 bars | Exponential weighted mean, α=2/(9+1) | 27 bars |
| EMA(12) | 12 bars | Exponential weighted mean, α=2/(12+1) | 36 bars |
| RSI(7) | 7 bars | EMA of gains / EMA of losses, 0-100 scale | 21 bars |
| VWAP deviation | 60 bars | (close - VWAP) / VWAP × 100 | 1 bar |
| Price momentum | 5 bars | (close[t] - close[t-5]) / close[t-5] × 100 | 5 bars |
| Volume momentum | 5 bars | (volume[t] - volume[t-5]) / volume[t-5] × 100 | 5 bars |

Total input features per bar: 1 (SPY close) + 1 (VIX close) + 7 (indicators) = **9 features**

### Training Configuration (defaults in config.py)

| Hyperparameter | Value | Rationale |
|---|---|---|
| sequence_length | 60 | 5-minute lookback window |
| prediction_steps | 5 | 25-second forecast horizon |
| n_features | 9 | SPY close + VIX close + 7 indicators |
| d_model | 128 | Balanced capacity vs. latency |
| n_heads | 4 | 32-dim head size, efficient on MPS and CPU |
| n_encoder_layers | 3 | Sufficient depth; ≥4 risks >100ms |
| ffn_dim | 512 | 4× d_model (standard) |
| dropout_rate | 0.2 | Regularization for small dataset |
| learning_rate | 3e-4 | AdamW default for transformers |
| batch_size | 256 | Large batch for stable gradient estimates |
| max_epochs | 1000 | Unconstrained; early stopping governs |
| early_stopping_patience | 30 | Patient enough for LR scheduling to help |
| train_val_test_split | 0.8 / 0.1 / 0.1 | Chronological, by trading day |
| lr_scheduler | CosineAnnealingWarmRestarts | Avoids local minima |
| random_seed | 42 | Reproducibility |

### Contracts

No REST API — this is a local script/library. The public interface contracts are function signatures:

**`preprocess.py`**
```
Input:  raw data directory path
Output: processed tensors saved to data/processed/ (train.pt, val.pt, test.pt)
        each file: dict with keys "sequences" (N, 60, 9) and "targets" (N, 5)
```

**`train.py`**
```
Input:  processed data path (optional: resume checkpoint path)
Output: best model checkpoint saved to models/
        experiment result appended to results/experiment_log.json
```

**`evaluate.py`**
```
Input:  checkpoint path, processed test data path
Output: per-step and aggregate metrics printed to stdout
        experiment result appended to results/experiment_log.json
        latency measured and logged per inference call
```

**`src/model/spy_predictor.py :: SpyPredictor.predict()`**
```
Input:  raw_input_window: np.ndarray shape (60, 9), un-normalized
Output: predicted_changes: np.ndarray shape (5,), percent change from bar 60's close
        directional_signal: str "up" | "down"
        latency_ms: float
```

---

## Complexity Tracking

No constitution violations. All design choices are justified by the spec requirements.

---

## Quickstart

After implementation, the development workflow is:

```bash
PYTHON=/Users/abeadam/dev/model/Stock-Model/.venv/bin/python

# 0. One-time symlink setup (data/raw → source directory, never copied)
ln -sfn "/Users/abeadam/dev/interactive-broker-python/Updated Stats/daily_data" data/raw

# 1. Pre-process all raw bar data (one-time, or re-run when new data arrives)
$PYTHON scripts/preprocess.py

# 2. Train
$PYTHON scripts/train.py

# 3. Evaluate best checkpoint on test set
$PYTHON scripts/evaluate.py --checkpoint models/spy_predictor_best.pth

# 4. Run full test suite
$PYTHON -m pytest tests/ -v

# 5. Run single test file
$PYTHON -m pytest tests/test_spy_predictor.py -v
```
