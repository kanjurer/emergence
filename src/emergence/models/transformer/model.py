import torch
import torch.nn as nn

from emergence.models.transformer.config import TransformerConfig
from emergence.models.transformer.embeddings import TokenAndPositionEmbedding
from emergence.models.transformer.block import TransformerBlock


class TransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.token_embeddings = TokenAndPositionEmbedding(config)

        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_layers)]
        )

        self.final_norm = nn.LayerNorm(config.embedding_dim)

        self.output_projection = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)

        x = self.blocks(x)

        x = self.final_norm(x)

        logits = self.output_projection(x)

        return logits
