# Tasks: SPY Intraday 5-Step Price Predictor

**Input**: Design documents from `specs/001-spy-predictor/`
**Prerequisites**: plan.md ✓, spec.md ✓, data-model.md ✓, research.md ✓

---

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story this task belongs to (US1–US4)
- **- [x]**: Already complete
- Exact file paths included in every task

---

## Phase 1: Setup

**Purpose**: Project structure, symlink, and dependency verification.

- [x] T001 Create project directory structure (`src/data/`, `src/model/`, `src/training/`, `src/evaluation/`, `src/utils/`, `scripts/`, `tests/`, `data/raw/`, `data/processed/`, `models/`, `results/`)
- [ ] T002 Create symlink `data/raw` → `/Users/abeadam/dev/interactive-broker-python/Updated Stats/daily_data` using `ln -sfn`
- [x] T003 [P] Implement `src/utils/config.py` — all hyperparameters, paths, MPS-first device selection
- [x] T004 [P] Implement `src/utils/logger.py` — structured stdout logging and `save_experiment_result()` appender
- [x] T005 [P] Implement `src/evaluation/metrics.py` — MAE, RMSE, R², directional accuracy (single-step functions)
- [x] T005b [P] Extend `src/evaluation/metrics.py` with `compute_per_step_metrics(actual: ndarray[N,5], predicted: ndarray[N,5]) → dict` — returns per-step keys `mae_step_{k}`, `rmse_step_{k}`, `r2_step_{k}`, `directional_accuracy_step_{k}` for k in 1..5 plus four aggregate keys; raises `ValueError` on non-2D or shape-mismatched input; add 7 new tests to `tests/test_metrics.py` covering all keys, perfect predictions, shape validation, and aggregate consistency (required by FR-006)
- [x] T006 [P] Write `tests/test_metrics.py` — 12 tests covering single-step metric functions; 7 additional tests for `compute_per_step_metrics` (added in T005b)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Data pipeline that all user stories depend on. No user story work starts until this phase is complete.

**⚠️ CRITICAL**: Phase 3+ cannot begin until T007–T015 are complete.

- [ ] T007 Implement `src/data/bar_loader.py` — load all `YYYY-MM-DD_SPY.txt` and `YYYY-MM-DD_VIX.txt` files from `data/raw/`, parse Unix epoch timestamps, sort bars by unix_timestamp ascending after loading (log a warning if out-of-order bars are detected), align SPY and VIX bars by timestamp (inner join per day), drop days with fewer than 60 aligned bars, return `dict[date, DataFrame]`
- [ ] T008 [P] Implement `src/data/indicator_computer.py` — compute EMA(5), EMA(9), EMA(12), RSI(7), VWAP deviation (60-bar), price momentum (5-bar), volume momentum (5-bar) from an aligned daily DataFrame; single-pass NumPy implementations; no external TA libraries
- [ ] T009 [P] Implement `src/data/normalizer.py` — per-feature z-score normalization over a 60-bar window; epsilon=1e-6; raises `ValueError` on NaN/Inf output; no inverse transform needed for targets (percent changes are the final prediction)
- [ ] T010 Implement `src/data/sequence_builder.py` — stride-1 sliding window over each day's feature matrix producing `(InputSequence, target)` pairs; windows must not span day boundaries; skip days with insufficient bars after indicator warmup; uses T008 and T009
- [ ] T011 [P] Implement `src/data/dataset.py` — `SpyDataset(torch.utils.data.Dataset)` wrapping processed sequences and targets; chronological train/val/test split by trading day (80/10/10); returns `(features_tensor, targets_tensor)` pairs
- [ ] T012 Implement `scripts/preprocess.py` — one-time pipeline: load raw bars (T007) → compute indicators (T008) → build sequences (T010) → split (T011) → save `data/processed/train.pt`, `val.pt`, `test.pt`; logs row counts and date range on completion
- [ ] T013 [P] Write `tests/test_bar_loader.py` — verify file discovery, timestamp parsing, SPY+VIX alignment, dropping of short days, deterministic output
- [ ] T014 [P] Write `tests/test_indicator_computer.py` — verify EMA convergence, RSI bounds (0–100), VWAP deviation sign, momentum sign; test edge cases (flat price, zero volume)
- [ ] T015 [P] Write `tests/test_sequence_builder.py` — verify no cross-day windows, correct input shape `(60, 9)`, correct target shape `(5,)`, no NaN values in output

**Checkpoint**: Data pipeline complete — user story implementation can now begin in parallel.

---

## Phase 3: User Story 1 — Fast Inference (Priority: P1) 🎯 MVP

**Goal**: Load a saved model and return 5 predicted SPY close prices + direction in under 100ms, with latency logged every call.

**Independent Test**: Run `scripts/evaluate.py --checkpoint models/spy_predictor_best.pth` against `data/processed/test.pt`. Confirm 5 predictions returned per sample, latency logged, no violations on MPS.

### Implementation for User Story 1

- [ ] T016 [P] [US1] Implement `src/model/attention.py` — `MultiHeadSelfAttention` using `torch.nn.functional.scaled_dot_product_attention`; supports `d_model`, `n_heads`; no custom CUDA kernels; moves to `COMPUTE_DEVICE` from config
- [ ] T017 [P] [US1] Implement `src/model/encoder_block.py` — `TransformerEncoderBlock`: self-attention → LayerNorm → residual → FFN(GELU) → LayerNorm → residual; dropout configurable from config
- [ ] T018 [US1] Implement `src/model/spy_predictor.py` — `SpyPredictor(nn.Module)`: linear input projection → sinusoidal positional encoding → N encoder blocks → last time-step → linear(d_model → prediction_steps); `predict(raw_window)` method returns `(predicted_changes: np.ndarray[5], directional_signal: str, latency_ms: float)`; `directional_signal = "up" if predicted_changes[-1] > 0.0 else "down"`; runs in `torch.inference_mode()`; uses `COMPUTE_DEVICE`
- [ ] T019 [P] [US1] Implement `src/model/checkpoint.py` — `save_checkpoint(model, config, val_loss, epoch, history, path)` and `load_checkpoint(path, device)` → returns `(SpyPredictor, TrainingConfig, metadata)`; validates `n_features` match on load; raises `ValueError` on mismatch
- [ ] T020 [P] [US1] Implement `src/evaluation/latency_tracker.py` — `measure_inference_latency(predict_fn, input_sequence)` → wraps predict call with `time.perf_counter`, returns `(prediction, latency_ms, is_violation)`; `is_violation = latency_ms > config.max_inference_latency_ms`; logs violation warning if exceeded
- [ ] T021 [P] [US1] Write `tests/test_spy_predictor.py` — verify output shape `(5,)`, deterministic outputs for same input, no NaN in output, directional signal is "up" or "down", directional signal is "up" when all outputs are biased positive (> 0.0), model moves to correct device, `ValueError` raised on short input
- [ ] T022 [P] [US1] Write `tests/test_latency_tracker.py` — verify latency is measured and returned, violation flag set correctly above threshold, no-violation flag below threshold

**Checkpoint**: US1 complete — inference pipeline independently functional and testable.

---

## Phase 4: User Story 2 — Best-Quality Training (Priority: P2)

**Goal**: Train on all available data to best validation loss; save best checkpoint; log all metrics.

**Independent Test**: Run `scripts/train.py`. Confirm `models/spy_predictor_best.pth` is created, validation loss decreases across early epochs, experiment log has one new entry with all metrics.

### Implementation for User Story 2

- [ ] T023 [P] [US2] Implement `src/training/scheduler.py` — `build_lr_scheduler(optimizer, config)` using `CosineAnnealingWarmRestarts`; configurable from `TrainingConfig`
- [ ] T024 [US2] Implement `src/training/trainer.py` — `Trainer` class: training loop with AdamW optimizer, per-epoch train/val loss tracking, early stopping on val loss (patience from config), saves best checkpoint via `checkpoint.py`, logs progress per epoch; supports resume from checkpoint path; uses `COMPUTE_DEVICE`
- [ ] T025 [US2] Implement `scripts/train.py` — entry point: load processed data → instantiate `SpyPredictor` and `Trainer` → train → compute test-set metrics via `metrics.py` → append full result to experiment log via `logger.py`; accepts optional `--resume` flag for checkpoint path
- [ ] T026 [P] [US2] Write `tests/test_trainer.py` — verify early stopping triggers after patience epochs, best checkpoint corresponds to lowest val loss epoch (not final epoch), experiment log entry is appended after training, training is deterministic given same seed

**Checkpoint**: US2 complete — full train → checkpoint → log pipeline functional.

---

## Phase 5: User Story 3 — Standalone Evaluation (Priority: P3)

**Goal**: Evaluate any saved checkpoint against the test set; report per-step and aggregate metrics; log result; measure and log inference latency.

**Independent Test**: Run `scripts/evaluate.py --checkpoint models/spy_predictor_best.pth`. Confirm per-step MAE/RMSE/R²/directional accuracy printed for steps 1–5, aggregate metrics printed, latency stats printed, experiment log has one new entry.

### Implementation for User Story 3

- [ ] T027 [US3] Implement `scripts/evaluate.py` — entry point: load checkpoint (T019) → load test data → run inference on each test sample via `latency_tracker.py` (T020) → compute per-step and aggregate metrics (T005) → print full report → append experiment result with latency stats via `logger.py`; accepts `--checkpoint` and `--data` args
- [ ] T028 [P] [US3] Write `tests/test_evaluate.py` — verify per-step metrics dict has all 5 steps, aggregate metrics present, latency stats (mean, max, violation count) present in logged result, two consecutive runs produce two separate log entries

**Checkpoint**: US3 complete — standalone evaluation independently functional.

---

## Phase 6: User Story 4 — Experiment History (Priority: P4)

**Goal**: Persistent, append-only experiment log readable after any number of runs.

**Independent Test**: Run training and evaluation twice each. Read `results/experiment_log.json`. Confirm 4 entries, each with timestamp + config + metrics, no entry modified or deleted.

### Implementation for User Story 4

- [ ] T029 [US4] Write `tests/test_logger.py` — verify append-only behavior (existing entries never modified), all required fields present in each record (timestamp, experiment_name, config, metrics), records in chronological order, file created if it doesn't exist, concurrent writes handled safely

**Checkpoint**: US4 complete — full experiment history guaranteed.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T030 [P] Build C++ indicator extension using pybind11 + Clang — implement `src/data/fast_indicators.cpp` with SIMD-vectorized single-pass EMA, RSI, VWAP deviation, and momentum computations; write `src/data/setup_indicators.py` (setuptools build script targeting Clang); expose as `fast_indicators` Python module; `indicator_computer.py` imports it at runtime with NumPy fallback if the extension is not built
- [ ] T031 [P] Enable `torch.compile()` on `SpyPredictor` in `scripts/train.py` and `scripts/evaluate.py` — wrap model with `torch.compile(model, mode="reduce-overhead")` after loading; skip gracefully on Python versions that don't support it
- [ ] T032 [P] Write `tests/test_normalizer.py` — verify z-score output has mean≈0 and std≈1 per feature, inverse transform recovers original values within float tolerance, `ValueError` raised on NaN/Inf input
- [ ] T033 [P] Write `tests/test_fast_indicators.py` — verify C++ extension output matches NumPy fallback output within float32 tolerance for EMA, RSI, VWAP deviation, momentum; test that build failure falls back gracefully to NumPy
- [ ] T034 Run full test suite and confirm all tests pass in under 60 seconds: `/Users/abeadam/dev/model/Stock-Model/.venv/bin/python -m pytest tests/ -v`
- [ ] T035 Create symlink `data/raw` (T002) and run `scripts/preprocess.py` end-to-end to verify pipeline against real data; confirm `data/processed/train.pt`, `val.pt`, `test.pt` created with correct shapes
- [ ] T036 Run a short training smoke-test (max_epochs=5) to verify train → checkpoint → evaluate → log pipeline works end-to-end on real data
- [ ] T037 Review all generated code against constitution Principle IV: descriptive variable names, single-responsibility modules, no magic numbers — fix any violations before closing

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — BLOCKS Phase 3, 4, 5, 6
- **Phase 3 (US1 — Inference)**: Depends on Phase 2 — no dependency on US2/US3/US4
- **Phase 4 (US2 — Training)**: Depends on Phase 2 and Phase 3 (needs `SpyPredictor`)
- **Phase 5 (US3 — Evaluation)**: Depends on Phase 3 and Phase 4 (needs model + trainer)
- **Phase 6 (US4 — History)**: Depends on Phase 1 (`logger.py` already written)
- **Phase 7 (Polish)**: Depends on all user story phases complete

### Within Phase 3 (US1)

```
T016, T017 run in parallel (separate files, no deps)
T018 depends on T016, T017
T019, T020 run in parallel after T018
T021, T022 run in parallel after T019, T020
```

### Parallel Opportunities Per Phase

```bash
# Phase 2 — run in parallel after T007:
pytest tests/test_bar_loader.py &
pytest tests/test_indicator_computer.py &
pytest tests/test_sequence_builder.py &

# Phase 3 — run in parallel:
# T016 and T017 together, then T018, then T019+T020+T021+T022 together

# Phase 7 — run in parallel:
# T030, T031, T032 simultaneously
```

---

## Implementation Strategy

### MVP (User Story 1 Only)

1. Complete Phase 1 (Setup)
2. Complete Phase 2 (Foundational) — data pipeline
3. Complete Phase 3 (US1 — Inference)
4. **STOP AND VALIDATE**: run `tests/`, confirm inference pipeline works independently
5. You now have a model you can load and call `.predict()` on

### Full Incremental Delivery

1. Setup → Foundational → **US1 inference** (MVP: predict from pre-trained weights)
2. → **US2 training** (can now train a model end-to-end)
3. → **US3 evaluation** (can now compare models)
4. → **US4 history** (already largely complete via logger)
5. → **Polish** (compile, benchmark, final test run)

---

## Notes

- Tasks marked `- [x]` are already implemented (`config.py`, `logger.py`, `metrics.py`, `test_metrics.py`)
- `[P]` tasks have no file conflicts and can be worked in parallel
- Sequence windows must never span day boundaries — enforced in `sequence_builder.py`
- `COMPUTE_DEVICE` from config must be used everywhere; never call `torch.device()` inline
- Every task that touches model or data code must be followed by a test run before marking done
- After T035 (code review), mark the task done only if all constitution principles pass
