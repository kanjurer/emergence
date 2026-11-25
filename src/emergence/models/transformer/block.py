import torch
import torch.nn as nn

from emergence.models.transformer.config import TransformerConfig
from emergence.models.transformer.attention import CausalSelfAttention
from emergence.models.transformer.mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        # Layer normalization
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.ln2 = nn.LayerNorm(config.embedding_dim)

        # Attention and MLP
        self.attention = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + residual connection
        x = x + self.attention(self.ln1(x))

        # MLP + residual connection
        x = x + self.mlp(self.ln2(x))

        return x
