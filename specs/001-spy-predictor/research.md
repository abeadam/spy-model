# Research: SPY Intraday Prediction Model

**Branch**: `001-spy-predictor` | **Date**: 2026-02-18

---

## 1. Model Architecture: Encoder-Only vs Encoder-Decoder

**Decision**: Encoder-only transformer with a direct projection head outputting 5 values.

**Rationale**: For a 5-step prediction horizon, encoder-decoder adds sequential decoding overhead with no accuracy benefit. Encoder-only eliminates the decoder bottleneck, giving 30-40% lower inference latency. The full 60-bar context window is processed in one pass; the last hidden state (or a learned aggregation) is projected directly to the 5 output prices.

**Alternatives considered**: Full encoder-decoder (SpaceTimeFormer style) — more expressive but slower and unnecessary for a 5-step horizon. RNN/LSTM — faster inference but weaker at capturing long-range dependencies across 60 bars.

---

## 2. Attention Mechanism

**Decision**: `torch.scaled_dot_product_attention` (PyTorch 2.0+) — full attention across all 60 tokens.

**Rationale**: At 60 tokens, O(n²) cost is ~3,600 operations — negligible. Flash Attention's memory savings are irrelevant at this sequence length. Windowed attention (as used in SpaceTimeFormer for thousands of tokens) would add management overhead that exceeds any savings. `scaled_dot_product_attention` auto-selects the optimal backend (Flash Attention on GPU, optimized kernels on CPU/MPS) and composes cleanly with `torch.compile()`.

**Alternatives considered**: Windowed attention — overkill for 60 tokens. Explicit Flash Attention package — unnecessary, already built into PyTorch 2.3+.

---

## 3. Inference Speed Strategy

**Decision**: `torch.compile()` during development; ONNX export for production.

**Rationale**: `torch.compile()` yields ~1.35× speedup on CPU with minimal code change, suitable for the development/research loop. ONNX + ONNXRuntime gives an additional 15-25% reduction over compiled PyTorch, bringing a d_model=128 model well under the 100ms budget. On Apple MPS (development machine), prioritize `torch.compile()` as ONNX MPS support is less mature.

**Latency estimates** (d_model=128, 3 layers, 4 heads):
- Baseline PyTorch: ~80-100ms
- With `torch.compile()`: ~55-70ms
- ONNX + ONNXRuntime: ~35-50ms

**Alternatives considered**: TorchScript — older, less effective than `torch.compile()`. Custom CUDA kernels — overkill for a 60-token sequence.

---

## 4. Model Size / Hyperparameters

**Decision**: Start at `d_model=128, n_heads=4, n_layers=3, ffn_dim=512`. Scale to `d_model=256, n_heads=8, n_layers=4` if validation loss plateaus.

**Rationale**: At d_model=128, 4 heads → 32-dim head size (efficient on both MPS and CPU). 3 layers captures sufficient depth for 60-step context. FFN at 4× d_model (512) is standard. Estimated ~105K parameters — small enough for fast inference on MPS, large enough to learn 5-second price dynamics. Do not exceed d_model=256 / 4 layers to stay well under the 100ms budget even on CPU fallback.

| Config | Params | ONNX latency (est.) |
|---|---|---|
| d=128, h=4, L=3 | ~595K | MPS ~10-25ms ✓ |
| d=256, h=8, L=4 | ~2.2M | MPS ~20-40ms ✓ |
| d=512, h=8, L=4 | ~8.5M | MPS ~40-70ms ✓ |

**Alternatives considered**: d=64 — too small for learning rich feature interactions. d=512 — hits latency ceiling.

---

## 5. C/C++ Extension Usage

**Decision**: C/C++ extension (pybind11) for technical indicator computation, if profiling shows NumPy/Numba insufficient.

**Rationale**: Indicator computation (EMA, RSI, VWAP deviation) is the most CPU-bound, vectorizable, non-PyTorch component of the pipeline. A pybind11 + SIMD (AVX2/NEON) extension can compute rolling indicators 3-5× faster than NumPy. However — indicators are pre-computed offline for training; the extension is only on the inference hot path. If all indicators are pre-computed and served from memory at inference time, C++ may not be needed.

**Priority order**:
1. Pre-compute indicators offline and cache to `data/processed/` → avoids C++ entirely.
2. If inference-time indicator computation is required (live data), use Numba JIT first (1.5-2.5× speedup, no compilation step).
3. If Numba is insufficient, write pybind11 C++ extension.

**Alternatives considered**: Custom CUDA kernels for attention — not justified at 60 tokens. TA-Lib — requires C build toolchain but no custom logic control.

---

## 6. Technical Indicators (Selected Set)

**Decision**: EMA(5), EMA(9), EMA(12), RSI(7), VWAP deviation (60-bar rolling), price momentum (5-bar), volume momentum (5-bar).

**Rationale**: All periods are computable within the 60-bar window with proper warmup. EMA(≤15) converges reliably within 60 bars. RSI(7) is optimal for high-frequency contexts (RSI(14) needs 42 bars to stabilize — too close to the window limit). VWAP deviation over the full window captures mean-reversion signal. Longer-period indicators (e.g. SMA(50)) would bias toward early bars and provide stale signal for a 25-second horizon.

**Alternatives considered**: RSI(14) — too long for 60-bar window. MACD — requires EMA(26), too long. Bollinger Bands — useful but periods >20 degrade at window edge.

---

## 7. Feature Normalization

**Decision**: Per-feature z-score normalization computed over each 60-bar input window (online, at inference time).

**Rationale**: Independent per-feature normalization prevents high-magnitude features (e.g. volume in thousands) from dominating low-magnitude ones (e.g. RSI 0-100). Computing stats over the current window (rather than fixed training-set statistics) adapts to current market regime (volatility, volume level). A small epsilon (1e-6) prevents division by zero in flat-price periods.

**Alternatives considered**: Fixed training-set mean/std — causes distribution shift during inference in different market regimes. Min-max scaling — unstable with outliers (price spikes, volume surges).

---

## 8. Data Split Strategy

**Decision**: Chronological split — no shuffling across time boundaries. Train on earliest 80%, validate on next 10%, test on final 10%. Within a day, sequence windows respect the day boundary (no cross-day sequences).

**Rationale**: Shuffling across time introduces look-ahead bias — the model would "know" future bars during training. Chronological splits simulate the real deployment scenario: the model is always predicting into unseen future data.

**304 trading days split**:
- Train: days 1–243 (~243 days)
- Validation: days 244–273 (~30 days)
- Test: days 274–304 (~30 days)
