from dataclasses import dataclass


@dataclass
class TransformerConfig:
    # Model architecture
    vocab_size: int = 256
    context_length: int = 128
    embedding_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4

    # Regularization
    dropout: float = 0.1

    def __post_init__(self):
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if self.context_length <= 0:
            raise ValueError("context_length must be positive")

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
