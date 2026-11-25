import torch
from emergence.models.transformer.config import TransformerConfig


class TokenAndPositionEmbedding(torch.nn.Module):
    token_embedding: torch.nn.Embedding
    position_embedding: torch.nn.Embedding

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.token_embedding = torch.nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
        )

        self.position_embedding = torch.nn.Embedding(
            config.context_length,
            config.embedding_dim,
        )

        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[1]

        if sequence_length > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds "
                f"context length ({self.position_embedding.num_embeddings})"
            )

        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(
            torch.arange(
                sequence_length,
                device=token_ids.device,
            ).unsqueeze(0)
        )

        embeddings = token_embeddings + position_embeddings

        return self.dropout(embeddings)
