<!-- SYNC IMPACT REPORT
Version change: (none) → 1.0.0  (initial ratification)
Added sections:
  - Core Principles (I–V)
  - Technology Stack
  - Development Workflow
  - Governance
Templates updated:
  - .specify/templates/plan-template.md  ✅ (Constitution Check rows align with principles below)
  - .specify/templates/spec-template.md  ✅ (success criteria aligned with SC metrics below)
  - .specify/templates/tasks-template.md ✅ (test tasks mandatory per Principle III)
Deferred TODOs: none
-->

# spy-model Constitution

## Core Principles

### I. Inference-First Performance

Model evaluation and inference MUST be fast. Every component on the prediction path —
data preprocessing, the forward pass, and metric computation — is subject to speed review.
Training time is explicitly unconstrained; we accept slow training in exchange for the
best possible model. We MUST NOT sacrifice evaluation speed to simplify training code.

**Rationale**: The model's production value is determined by how quickly it can score new
data. A slow training run is acceptable; a slow evaluation loop is not.

### II. Train-to-Best

Training MUST pursue the highest achievable model quality without regard for wall-clock time.
Techniques such as extended epochs, learning-rate scheduling, ensemble methods, and exhaustive
hyperparameter search are all appropriate. Early-stopping MUST target validation loss, not
training time. We MUST NOT prematurely stop training to save time.

**Rationale**: A marginally better model compounds over many predictions. The cost of extra
training is compute; the cost of a suboptimal model is accuracy loss on every inference.

### III. Test-and-Track (NON-NEGOTIABLE)

Every task that modifies model architecture, training logic, data preprocessing, or evaluation
code MUST end with a passing test run. No task is complete until:

1. The relevant pytest suite passes with zero failures.
2. The evaluation result is appended to `results/experiment_log.json` with timestamp,
   config snapshot, and all metrics.

Results MUST be appended — never overwritten. The full history of experiments is a
first-class artifact of this project.

**Rationale**: Untested model changes silently regress performance. A persistent experiment
log is the only way to know whether a change is an improvement.

### IV. Staff-Level Code Quality

All code MUST meet staff software engineer standards:

- Variable names MUST be fully descriptive (`close_prices`, not `cp`; `hidden_size`, not `hs`).
- Each file MUST have a single, clear responsibility. Data loading, preprocessing, model
  architecture, training loop, evaluation, and configuration are always separate modules.
- No magic numbers or hardcoded paths anywhere outside `src/utils/config.py`.
- Functions MUST be short and do one thing. If a function requires an explanatory comment,
  it MUST be split or renamed instead.
- After completing any task, generated code MUST be reviewed and improved before the task
  is marked done.

**Rationale**: This codebase will be read and extended repeatedly. Clarity now prevents
bugs and wasted time later.

### V. Config-Driven Architecture

All hyperparameters, file paths, and tuneable settings MUST live exclusively in
`src/utils/config.py`. Model files, training scripts, and evaluation scripts MUST import
from config — never define their own constants. Changes to any hyperparameter require
only a single-file edit.

**Rationale**: Scattered constants make experiments unreproducible and hyperparameter
sweeps error-prone. Centralised config ensures every run is fully described by its
config snapshot.

## Technology Stack

- **Runtime**: Python 3.13 via `/Users/abeadam/dev/model/Stock-Model/.venv/bin/python`
- **ML framework**: PyTorch (GPU-capable via the shared venv)
- **Data**: pandas, numpy
- **Evaluation / metrics**: scikit-learn, custom `src/evaluation/metrics.py`
- **Visualisation**: matplotlib
- **Testing**: pytest (run via full venv path, not shell activation)
- **Test command**: `/Users/abeadam/dev/model/Stock-Model/.venv/bin/python -m pytest tests/ -v`

No dependency may be added without a clear, irreplaceable purpose.

## Development Workflow

1. **Config first** — any new hyperparameter goes into `src/utils/config.py` before
   any other file is touched.
2. **Test before and after** — run the suite before starting a task (confirm baseline),
   then again after (confirm no regression).
3. **Log every result** — call `save_experiment_result()` at the end of every training
   or evaluation run, including intermediate experiments.
4. **Review before done** — read the generated code critically. Ask: is every variable
   name clear? Does each function do exactly one thing? Is there simpler? Improve, then close.
5. **Keep files small and focused** — if a file grows beyond ~150 lines, consider whether
   it is handling more than one responsibility.

## Governance

This constitution supersedes all other conventions in the repository. Any amendment requires:

1. A clear rationale tied to an observed project need.
2. A version bump per semantic versioning (MAJOR for principle removal/redefinition,
   MINOR for new principle or section, PATCH for clarification).
3. An update to this file's Sync Impact Report comment and the `Last Amended` date.

All generated plans, specs, and task lists MUST reference these principles explicitly in
their Constitution Check section.

**Version**: 1.0.0 | **Ratified**: 2026-02-18 | **Last Amended**: 2026-02-18
