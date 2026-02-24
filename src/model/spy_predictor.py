"""
Full SPY Intraday Price Predictor model.

Architecture:
    raw 60-bar window (60, n_features)
        ↓  normalize_window()
    normalized input (60, n_features)
        ↓  Linear projection
    (batch, 60, d_model)
        ↓  + sinusoidal positional encoding
    (batch, 60, d_model)
        ↓  TransformerEncoderBlock × n_encoder_layers
    (batch, 60, d_model)
        ↓  select last time-step
    (batch, d_model)
        ↓  Linear(d_model → prediction_steps)
    (batch, 5)  ←  5 predicted SPY close percent changes from bar 60

The last time-step position aggregates the full bidirectional context of
all 60 bars (encoder-only, full attention). This is faster and simpler than
pooling across all 60 positions.

Positional encoding is sinusoidal (fixed, no learned parameters) — sufficient
for a 60-position sequence and eliminates a learned embedding matrix.
"""

import time

import numpy as np
import torch
import torch.nn as nn

from src.data.normalizer import normalize_window
from src.model.encoder_block import TransformerEncoderBlock
from src.utils.config import COMPUTE_DEVICE, DEFAULT_CONFIG, TrainingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_sinusoidal_position_encoding(
    sequence_length: int,
    d_model: int,
) -> torch.Tensor:
    """
    Build a fixed sinusoidal positional encoding matrix.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Returns a tensor of shape (1, sequence_length, d_model) — broadcastable
    over the batch dimension.
    """
    positions  = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
    dimensions = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term   = torch.pow(10_000.0, dimensions / d_model)

    encoding          = torch.zeros(sequence_length, d_model)
    encoding[:, 0::2] = torch.sin(positions / div_term)
    encoding[:, 1::2] = torch.cos(positions / div_term)

    return encoding.unsqueeze(0)  # (1, sequence_length, d_model)


class SpyPredictor(nn.Module):
    """
    Encoder-only transformer for SPY intraday 5-step price prediction.

    Parameters
    ----------
    config : TrainingConfig
        Supplies all architectural hyperparameters.
    device : torch.device
        Compute device. Use COMPUTE_DEVICE from config (MPS → CUDA → CPU).
    """

    def __init__(
        self,
        config: TrainingConfig = DEFAULT_CONFIG,
        device: torch.device = COMPUTE_DEVICE,
    ) -> None:
        super().__init__()
        self.config         = config
        self._compute_device = device

        self.input_projection = nn.Linear(config.n_features, config.d_model)

        self.register_buffer(
            "positional_encoding",
            _build_sinusoidal_position_encoding(config.sequence_length, config.d_model),
        )

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                ffn_dim=config.ffn_dim,
                dropout_rate=config.dropout_rate,
            )
            for _ in range(config.n_encoder_layers)
        ])

        self.output_projection = nn.Linear(config.d_model, config.prediction_steps)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, sequence_length, n_features). Normalized input.

        Returns
        -------
        torch.Tensor
            Shape (batch_size, prediction_steps). Predicted SPY close percent changes.
        """
        # Project features to model dimension.
        x = self.input_projection(x)                  # (batch, seq, d_model)

        # Add fixed sinusoidal positional encoding (broadcast over batch).
        x = x + self.positional_encoding              # (batch, seq, d_model)

        # Pass through encoder blocks.
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)                      # (batch, seq, d_model)

        # Select the last time-step — it aggregates full bidirectional context.
        last_hidden_state = x[:, -1, :]               # (batch, d_model)

        return self.output_projection(last_hidden_state)  # (batch, prediction_steps)

    def predict(
        self,
        raw_input_window: np.ndarray,
    ) -> tuple[np.ndarray, str, float]:
        """
        Run end-to-end inference on a single raw (un-normalized) input window.

        Normalizes the input internally, runs the forward pass under
        torch.inference_mode(), and returns percent-change predictions directly
        (no denormalization needed — the model learns to predict returns).

        Parameters
        ----------
        raw_input_window : np.ndarray
            Shape (sequence_length, n_features). Raw feature values (not normalized).

        Returns
        -------
        predicted_changes : np.ndarray
            Shape (prediction_steps,) float32. Predicted SPY close percent changes
            relative to bar 60's close (e.g. 0.002 = +0.2%).
        directional_signal : str
            "up" if predicted_changes[-1] > 0.0 else "down".
        latency_ms : float
            End-to-end wall-clock inference time in milliseconds.

        Raises
        ------
        ValueError
            If raw_input_window has wrong shape or contains NaN / Inf values.
        """
        expected_shape = (self.config.sequence_length, self.config.n_features)
        if raw_input_window.shape != expected_shape:
            raise ValueError(
                f"raw_input_window must have shape {expected_shape}. "
                f"Got {raw_input_window.shape}."
            )

        inference_start = time.perf_counter()

        # normalize_window raises ValueError on NaN/Inf.
        normalized_window, _, _ = normalize_window(raw_input_window)

        input_tensor = (
            torch.from_numpy(normalized_window)
            .unsqueeze(0)
            .to(self._compute_device)
        )  # (1, sequence_length, n_features)

        with torch.inference_mode():
            model_output = self(input_tensor)                   # (1, prediction_steps)
            predicted_changes = model_output.squeeze(0).cpu().numpy().astype(np.float32)

        directional_signal = "up" if float(predicted_changes[-1]) > 0.0 else "down"
        latency_ms = (time.perf_counter() - inference_start) * 1_000.0

        return predicted_changes, directional_signal, latency_ms

    def parameter_count(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
