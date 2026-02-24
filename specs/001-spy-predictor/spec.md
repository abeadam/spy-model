# Feature Specification: SPY Intraday Price Prediction Model

**Feature Branch**: `001-spy-predictor`
**Created**: 2026-02-18
**Status**: Draft
**Input**: SPY intraday price prediction model using 5-second bars, predicting the next 5 bar percent-change returns, with fast inference, best-quality training, experiment tracking, and full test coverage

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Get the Next 5 Bar Price Predictions (Priority: P1)

A trader provides a sequence of recent SPY 5-second bars and receives predicted closing
prices for the next 5 bars (covering the next 25 seconds of trading). Predictions are
returned fast enough to be acted on before the next bar closes.

**Why this priority**: This is the primary value the model delivers. Without a working,
fast prediction path, nothing else in the system matters.

**Independent Test**: Load a saved model and a valid sequence of recent 5-second bars.
Call the prediction function. Confirm 5 predicted percent-change returns are returned
within the latency limit.

**Acceptance Scenarios**:

1. **Given** a trained model and a valid sequence of recent SPY 5-second bars,
   **When** the prediction function is called,
   **Then** it returns exactly 5 predicted closing-price percent changes (one per future
   bar, relative to bar 60's close) within 100ms.

2. **Given** a trained model,
   **When** the same input sequence is evaluated twice,
   **Then** both sets of predictions are identical (deterministic inference).

3. **Given** an input sequence shorter than the required window length,
   **When** the prediction function is called,
   **Then** a clear error is returned describing the shortfall, and no prediction is made.

4. **Given** predictions for bars 1 through 5,
   **When** a directional signal is derived,
   **Then** the signal reflects whether bar 5's predicted percent change is positive
   (up) or non-positive (down), indicating net direction over the 25-second window.

---

### User Story 2 — Train the Best Possible Model (Priority: P2)

A researcher runs training on all available historical SPY intraday bar data and receives
a trained model that achieves the best achievable multi-step prediction accuracy on
held-out validation data. Training time is not a constraint.

**Why this priority**: Prediction quality (US1) is entirely determined by training quality.
Training is the only place where model accuracy can be improved.

**Independent Test**: Run the training script against the full dataset. Confirm a model
artifact is saved, validation loss decreases over training, and all metrics are logged.

**Acceptance Scenarios**:

1. **Given** all available SPY 5-second bar data (304 trading days),
   **When** training is initiated,
   **Then** the system trains until the best validation loss is achieved, saves the best
   checkpoint, and appends a complete result record to the experiment log.

2. **Given** a training run where validation loss begins rising while training loss falls,
   **When** the early stopping patience is exhausted,
   **Then** the saved model is from the best validation checkpoint, not the final epoch.

3. **Given** a completed training run,
   **When** training finishes,
   **Then** a record with timestamp, full configuration snapshot, and all evaluation
   metrics is appended to the experiment log — no prior records are modified.

---

### User Story 3 — Evaluate a Saved Model on New Data (Priority: P3)

A researcher evaluates any saved model against held-out test bars and receives a full
multi-step performance report without triggering retraining.

**Why this priority**: Decoupled evaluation enables rapid model comparison without
wasting training compute.

**Independent Test**: Point the evaluation script at a saved model and a test set.
Confirm a metrics report is printed per prediction step and an entry is appended to
the experiment log.

**Acceptance Scenarios**:

1. **Given** a saved model and a held-out test set of 5-second bars,
   **When** evaluation is run,
   **Then** the system reports MAE, RMSE, R², and directional accuracy for each of
   the 5 prediction steps and as an aggregate.

2. **Given** two saved models evaluated on the same test set,
   **When** the experiment log is inspected,
   **Then** they appear as two independent entries that can be compared by metrics
   and configuration.

---

### User Story 4 — Review Experiment History (Priority: P4)

A researcher inspects the persistent experiment log to understand how model accuracy
has evolved across all training runs and evaluations.

**Why this priority**: Without a full history, there is no way to know whether a change
improved or degraded performance.

**Independent Test**: After multiple runs, inspect the log. Confirm each run is a
separate, immutable record in chronological order.

**Acceptance Scenarios**:

1. **Given** multiple completed runs,
   **When** the experiment log is inspected,
   **Then** every run is a separate record with timestamp, config snapshot, and
   per-step and aggregate metrics.

2. **Given** an existing log,
   **When** a new run completes,
   **Then** exactly one new record is appended; no existing record is modified.

---

### Edge Cases

- What happens when a day's raw file contains missing bars (e.g. halts, pre/post market gaps)?
- How does the system handle bar timestamps that are out of chronological order within a file?
- What if the raw data file for a trading day is missing entirely?
- What if a training run is interrupted — is the best checkpoint saved to that point recoverable?
- How are sequence boundaries handled at the start and end of a trading day
  (e.g. a prediction window that spans the market close)?
- What if the model predicts a price outside any historically plausible range?

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST accept a fixed-length sequence of SPY 5-second OHLCV bars
  plus contemporaneous VIX bar data and derived technical indicators as input, and return
  exactly 5 predicted SPY closing-price percent changes covering the next 5 bars (next
  25 seconds), each relative to the last input bar's close. VIX bars are aligned to SPY
  bars by timestamp and treated as additional input features at each time step — VIX is
  not predicted by the model.
- **FR-002**: The system MUST produce a net directional signal (up or down) for the
  5-bar prediction window, derived from the sign of bar 5's predicted percent change:
  positive = "up", non-positive = "down".
- **FR-003**: The system MUST load all available SPY and VIX 5-second bar files from
  `/Users/abeadam/dev/interactive-broker-python/Updated Stats/daily_data/` and treat
  them as the complete training corpus. No manual data selection is permitted.
- **FR-004**: The system MUST train exclusively on chronologically ordered data —
  training bars MUST precede validation bars, which MUST precede test bars.
  No shuffling across the time boundary.
- **FR-005**: The system MUST save the model checkpoint with the best validation loss
  seen during training. The final epoch's weights are NOT used unless they happen to be
  the best.
- **FR-006**: The system MUST evaluate multi-step prediction performance using at minimum:
  MAE, RMSE, R², and directional accuracy — reported per prediction step (steps 1–5)
  and as an aggregate across all steps.
- **FR-007**: The system MUST append a structured result record to the experiment log
  after every training run and every standalone evaluation. Records MUST include:
  timestamp, full configuration snapshot, and all per-step and aggregate metrics.
- **FR-008**: The system MUST normalize input features before the model using per-feature
  z-score normalization over the 60-bar input window. Model outputs are percent-change
  returns and do not require denormalization — they are the final prediction.
- **FR-009**: The system MUST include a test suite covering data loading, preprocessing,
  indicator computation, metric correctness, and model input/output shape. The suite
  MUST complete in under 60 seconds.
- **FR-010**: All hyperparameters (sequence length fixed at 60 bars, prediction steps
  fixed at 5, indicator periods, model architecture settings, train/val/test split
  ratios) and file paths MUST be configurable from a single location without touching
  model or training code.
- **FR-011**: The system MUST measure and log the end-to-end evaluation latency — from
  the moment input data is provided to the moment predictions are returned — for every
  inference call made by the final deployed model. This latency MUST NOT exceed 100ms.
  Any evaluation that exceeds 100ms MUST be flagged as a performance violation in the log.

### Key Entities

- **BarRecord**: A single 5-second SPY trading bar — Unix timestamp, open, high, low,
  close, volume. The atomic unit of raw data.
- **FeatureVector**: A single time step's model input — SPY close price, the
  contemporaneous VIX close price, and derived technical indicators (moving averages,
  RSI, and similar) computed from surrounding SPY BarRecords. VIX is a raw co-feature;
  OHLCV fields beyond close are used only to compute SPY indicators.
- **InputSequence**: A fixed-length, chronologically ordered window of FeatureVectors
  used as one model input sample.
- **MultiStepPrediction**: A model output for one InputSequence — exactly 5 predicted
  closing-price percent changes relative to bar 60's close, plus a net directional signal.
- **ExperimentResult**: An immutable log entry for one training run or evaluation —
  timestamp, full configuration, and per-step plus aggregate metrics.
- **ModelCheckpoint**: A saved model artifact paired with the configuration and training
  metadata from the run that produced it.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The model achieves directional accuracy above 52% on the held-out test set
  for the 5-bar aggregate direction (better than a 50% random baseline).
- **SC-002**: A single 5-step prediction, given a pre-loaded model, completes in under
  100ms end-to-end. The measured latency for every inference call is recorded in the
  experiment log, and any call exceeding 100ms is flagged as a performance violation.
- **SC-003**: The full automated test suite completes in under 60 seconds.
- **SC-004**: Every training run and evaluation produces a logged result record containing
  per-step metrics for all 5 prediction steps. After 10 experiments, all 10 records are
  present in the experiment log with no duplicates or missing entries.
- **SC-005**: A researcher can identify the best-performing historical model configuration
  by inspecting the experiment log alone — no retraining required.
- **SC-006**: The model checkpoint saved at the end of training corresponds to the epoch
  with the lowest validation loss — verified by comparing checkpoint metrics to the
  training log.

### Assumptions

- The prediction horizon is exactly 5 bars (25 seconds). Longer horizons are out of scope.
- The input sequence window is exactly 60 bars (5 minutes of context).
- Each raw file contains bars for a single trading day and a single symbol,
  named `YYYY-MM-DD_SYMBOL.txt`, with a Unix epoch timestamp in the `Date` column.
- Sequence windows MUST NOT span day boundaries — each InputSequence contains bars
  from a single trading day only.
- The model is retrained manually. There is no automated retraining trigger.
- Data sourcing is out of scope. All SPY data files are pre-existing at
  `/Users/abeadam/dev/interactive-broker-python/Updated Stats/daily_data/`.
  304 trading days of SPY data are available (2024-11-04 through 2026-01-20).
- More data with same structure will be added in the future in the same directory.

---

## Clarifications

### Session 2026-02-18

- Q: What is the file format and required columns for raw input data? → A: Per-day per-symbol `.txt` files named `YYYY-MM-DD_SYMBOL.txt` containing 5-second intraday OHLCV bars with a Unix epoch timestamp in the `Date` column. 304 SPY trading days available (2024-11-04 to 2026-01-20).
- Q: Which fields does the model use as input features? → A: Closing price plus derived technical indicators (moving averages, RSI, and similar). All OHLCV fields are source data; indicators are computed during preprocessing.
- Q: How much historical data should be used? → A: All available data — full 304 trading days from the existing data directory.
- Q: How should intraday bars be used as model input? → A: Raw 5-second bars are the direct model input (no daily aggregation). The model predicts the next 5 bar closing prices (25-second multi-step horizon).
- Q: Which correlated symbols should be included as co-features? → A: VIX bars only. VIX close is aligned to SPY bars by timestamp and included as an additional feature at each time step. SPY is the only prediction target.
- Q: What is the input sequence window length? → A: 60 bars (5 minutes of 5-second bar context).
