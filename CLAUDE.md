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

- **One responsibility per file.** Data loading, preprocessing, model definition, training loop, and evaluation are always separate modules.
- **Config-driven.** All hyperparameters and paths live in `src/utils/config.py` — never hardcoded in training or model files.
- **Evaluation is fast.** The inference path (forward pass + metrics) must be optimized. Training can be slow; evaluation cannot.
- **Every task ends with a test.** After any model or data change, run the test suite. A task is not done until tests pass.
- **Results are always saved.** Every training run and evaluation appends a structured record to `results/experiment_log.json` including timestamp, config snapshot, and all metrics.

## Code Quality Standards

- Code quality must match staff-level engineering: clear abstractions, no magic numbers, no duplication.
- Variable names must be descriptive (`close_prices` not `cp`, `hidden_size` not `hs`).
- After completing any task, **review the generated code for clarity, correctness, and simplicity — then improve it** before marking the task done.
- Functions should be short and do one thing. If a function needs a comment to explain what it does, it should be split or renamed.
- Prefer explicit over clever.
