import torch

from emergence.models.transformer.config import TransformerConfig
from emergence.models.transformer.embeddings import TokenAndPositionEmbedding


def main():
    config = TransformerConfig()

    print("=== Transformer Configuration ===")
    print(config)

    model = TokenAndPositionEmbedding(config)

    tokens = torch.randint(
        0,
        config.vocab_size,
        (2, 10),
    )

    output = model(tokens)

    print("\n=== Embedding Experiment ===")
    print("Input shape:", tokens.shape)
    print("Output shape:", output.shape)


    # This is vocab_size*embedding_dim + context_length*embedding_dim
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    print("\nFirst token representation:")
    print(output[0, 0])


if __name__ == "__main__":
    main()
