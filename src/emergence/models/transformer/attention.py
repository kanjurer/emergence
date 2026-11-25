import torch
import torch.nn as nn
import torch.nn.functional as F

from emergence.models.transformer.config import TransformerConfig


class CausalSelfAttention(nn.Module):
    causal_mask: torch.Tensor

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads

        # Combined Query, Key, Value projection: we use same linear layer to generate query, key and value vectors
        self.qkv = nn.Linear(
            config.embedding_dim,
            3 * config.embedding_dim,
        )

        # Output projection
        self.output_projection = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
        )

        # Dropout
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        # Causal mask: prevents tokens from attending to future tokens (GPT-style)
        # This is an interesting detail
        causal_mask = torch.tril(
            torch.ones(
                config.context_length,
                config.context_length,
                dtype=torch.bool,
            )
        )

        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Split into attention heads
        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Calculate attention scores
        attention_scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply causal mask
        mask = self.causal_mask[:sequence_length, :sequence_length]

        attention_scores = attention_scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        # Convert scores into attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # Apply attention weights to values
        output = attention_weights @ v

        # Combine attention heads
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            embedding_dim,
        )

        # Final projection
        output = self.output_projection(output)

        return self.residual_dropout(output)
