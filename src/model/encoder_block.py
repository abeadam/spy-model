"""
Single transformer encoder block: self-attention + feed-forward network.

Block structure (Post-LN, as in the original "Attention Is All You Need"):
    x = LayerNorm(x + MultiHeadSelfAttention(x))
    x = LayerNorm(x + FFN(x))

The feed-forward network expands to ffn_dim then contracts back to d_model
using GELU activation (smooth, non-saturating, preferred over ReLU for
transformers on continuous-valued sequences).
"""

import torch
import torch.nn as nn

from src.model.attention import MultiHeadSelfAttention


class TransformerEncoderBlock(nn.Module):
    """
    One transformer encoder block with residual connections and layer norm.

    Parameters
    ----------
    d_model : int
        Input and output dimensionality.
    n_heads : int
        Number of attention heads passed to MultiHeadSelfAttention.
    ffn_dim : int
        Inner dimension of the feed-forward network (typically 4 × d_model).
    dropout_rate : float
        Dropout rate applied after attention and inside the FFN.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.self_attention  = MultiHeadSelfAttention(d_model, n_heads, dropout_rate)
        self.attention_norm  = nn.LayerNorm(d_model)
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.feed_forward_network = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_dim, d_model),
        )
        self.ffn_norm    = nn.LayerNorm(d_model)
        self.ffn_dropout = nn.Dropout(dropout_rate)

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
        # Self-attention sublayer with residual and Post-LN.
        attention_output = self.attention_dropout(self.self_attention(x))
        x = self.attention_norm(x + attention_output)

        # Feed-forward sublayer with residual and Post-LN.
        ffn_output = self.ffn_dropout(self.feed_forward_network(x))
        x = self.ffn_norm(x + ffn_output)

        return x
