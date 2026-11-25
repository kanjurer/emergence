import torch
import torch.nn as nn

from emergence.models.transformer.config import TransformerConfig


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.network = nn.Sequential(
            # Expand embedding dimension
            nn.Linear(
                config.embedding_dim,
                4 * config.embedding_dim,
            ),
            # Non-linear activation
            nn.GELU(),
            # Project back to original embedding dimension
            nn.Linear(
                4 * config.embedding_dim,
                config.embedding_dim,
            ),
            # Dropout
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
