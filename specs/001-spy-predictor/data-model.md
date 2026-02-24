# Data Model: SPY Intraday 5-Step Predictor

**Branch**: `001-spy-predictor` | **Date**: 2026-02-18

---

## Entities

### BarRecord
A single 5-second OHLCV observation for one symbol on one trading day.

| Field | Type | Constraints |
|---|---|---|
| unix_timestamp | int64 | Unix epoch seconds; unique per symbol per day |
| open | float32 | > 0 |
| high | float32 | ≥ open, ≥ close |
| low | float32 | ≤ open, ≤ close, > 0 |
| close | float32 | > 0 |
| volume | int64 | ≥ 0 |
| symbol | str | "SPY" or "VIX" |
| trading_date | date | Derived from unix_timestamp in US/Eastern timezone |

**Source**: `YYYY-MM-DD_SYMBOL.txt` files in raw data directory.
**Uniqueness**: (unix_timestamp, symbol) is the natural key.

---

### AlignedBarFrame
A merged view of SPY + VIX bars for a single trading day, aligned by timestamp.

| Field | Type | Notes |
|---|---|---|
| unix_timestamp | int64 | Timestamps present in both SPY and VIX files |
| spy_close | float32 | SPY close price at this bar |
| vix_close | float32 | VIX close price at this bar |

**Alignment rule**: Only timestamps present in BOTH the SPY and VIX files for that day are kept. Bars with no VIX counterpart are dropped. This ensures every row has all required co-features.

---

### FeatureVector
The computed model input for a single bar, after indicator computation and normalization.

| Field | Type | Range (pre-normalization) | Notes |
|---|---|---|---|
| spy_close | float32 | ~$400-600 | Raw close; normalized within window |
| vix_close | float32 | ~$10-80 | Raw VIX close; normalized within window |
| ema_5 | float32 | ~$400-600 | 5-bar EMA of spy_close |
| ema_9 | float32 | ~$400-600 | 9-bar EMA of spy_close |
| ema_12 | float32 | ~$400-600 | 12-bar EMA of spy_close |
| rsi_7 | float32 | 0-100 | 7-period RSI of spy_close |
| vwap_deviation | float32 | % | (spy_close - VWAP) / VWAP × 100; VWAP over 60-bar window |
| price_momentum_5 | float32 | % | (spy_close[t] - spy_close[t-5]) / spy_close[t-5] × 100 |
| volume_momentum_5 | float32 | % | (volume[t] - volume[t-5]) / volume[t-5] × 100 |

**Total features per bar**: 9
**Normalization**: Per-feature z-score computed over the 60-bar input window (mean=0, std=1, epsilon=1e-6).

---

### InputSequence
A 60-bar window of FeatureVectors used as one model input sample.

| Field | Type | Shape | Notes |
|---|---|---|---|
| features | float32 tensor | (60, 9) | Normalized; chronological order (oldest first) |
| day_boundary_flag | bool | — | Sequence must not span trading day boundaries |

**Construction**: Stride-1 sliding window over each day's AlignedBarFrame. The 60th bar is the most recent; bars 1-60 feed into the encoder.

**Minimum valid sequence**: Requires at least 60 bars in a single day with no gaps. Days with fewer than 60 bars after alignment are skipped.

---

### MultiStepPrediction
Output of one forward pass through the model.

| Field | Type | Notes |
|---|---|---|
| predicted_changes | float32[5] | Next 5 SPY close percent changes relative to bar 60's close (e.g. 0.002 = +0.2%) |
| directional_signal | str | "up" if predicted_changes[4] > 0.0 else "down" |
| latency_ms | float | End-to-end inference time in milliseconds |
| is_latency_violation | bool | True if latency_ms > 100.0 |

---

### ExperimentResult
An immutable log entry appended to `results/experiment_log.json` after every run.

| Field | Type | Notes |
|---|---|---|
| timestamp | str | ISO 8601 UTC timestamp |
| experiment_name | str | e.g. "training_run", "evaluation" |
| config | dict | Full snapshot of TrainingConfig.to_dict() |
| metrics | dict | Keys: mae_step_{1-5}, rmse_step_{1-5}, r2_step_{1-5}, directional_accuracy_step_{1-5}, mae_aggregate, rmse_aggregate, r2_aggregate, directional_accuracy_aggregate |
| latency_stats | dict | For evaluation runs: mean_latency_ms, max_latency_ms, violation_count, violation_rate |
| notes | str | Optional free-text annotation |

**Immutability rule**: Records are only appended. No record is ever modified or deleted.

---

### ModelCheckpoint
A saved `.pth` file containing the model with the best validation loss seen during a training run.

| Field (in checkpoint dict) | Type | Notes |
|---|---|---|
| model_state_dict | dict | PyTorch state dict |
| config | dict | TrainingConfig snapshot from that run |
| best_val_loss | float | Validation loss at the epoch this checkpoint was saved |
| best_epoch | int | Which epoch produced this checkpoint |
| training_history | dict | Per-epoch train_loss and val_loss lists |
| n_features | int | Must match input data at load time |

---

## State Transitions

```
Raw .txt files
    ↓  preprocess.py
Processed .pt tensors (train / val / test splits)
    ↓  train.py
ModelCheckpoint (.pth) + ExperimentResult in log
    ↓  evaluate.py
Metrics report (stdout) + ExperimentResult in log
    ↓  spy_predictor.predict()
MultiStepPrediction + ExperimentResult in log
```

---

## Validation Rules

- A BarRecord with `high < low` is invalid and must be dropped with a logged warning.
- An AlignedBarFrame with fewer than 60 bars after alignment is skipped; a warning is logged naming the date.
- An InputSequence where any feature value is NaN or Inf after normalization causes a ValueError — not silently passed to the model.
- A ModelCheckpoint loaded with a mismatched `n_features` raises a ValueError before any forward pass.
