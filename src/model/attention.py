"""
Multi-head self-attention using torch.nn.functional.scaled_dot_product_attention.

Full attention across all 60 tokens — at this sequence length the O(n²) cost
(3,600 operations) is negligible and Flash Attention's memory optimizations
are unnecessary. PyTorch's scaled_dot_product_attention auto-selects the
optimal backend (Flash Attention on GPU, optimized kernels on MPS/CPU).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention layer.

    Projects queries, keys, and values from the same input sequence
    (self-attention), splits into n_heads, applies scaled dot-product
    attention, and projects the concatenated heads back to d_model.

    Parameters
    ----------
    d_model : int
        Model dimensionality. Must be divisible by n_heads.
    n_heads : int
        Number of attention heads. head_dim = d_model // n_heads.
    dropout_rate : float
        Dropout applied to attention weights during training (0.0 at inference).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        self.d_model      = d_model
        self.n_heads      = n_heads
        self.head_dim     = d_model // n_heads
        self.dropout_rate = dropout_rate

        # Fused Q/K/V projection — one linear layer instead of three.
        self.qkv_projection    = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, sequence_length, d_model).

        Returns
        -------
        torch.Tensor
            Shape (batch_size, sequence_length, d_model).
        """
        batch_size, sequence_length, _ = x.shape

        # Project to Q, K, V in one shot, then split along last dim.
        qkv = self.qkv_projection(x)                    # (batch, seq, 3 * d_model)
        query, key, value = qkv.split(self.d_model, dim=-1)   # each: (batch, seq, d_model)

        # Reshape to (batch, n_heads, seq, head_dim) for SDPA.
        query = query.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        key   = key.view(  batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)

        attention_dropout = self.dropout_rate if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=attention_dropout
        )  # (batch, n_heads, seq, head_dim)

        # Merge heads: (batch, seq, d_model).
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.d_model
        )
        return self.output_projection(attended)
