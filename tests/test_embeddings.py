import torch

from emergence.models.transformer.config import TransformerConfig
from emergence.models.transformer.embeddings import TokenAndPositionEmbedding


def test_embedding_output_shape():
    config = TransformerConfig()
    model = TokenAndPositionEmbedding(config)

    tokens = torch.randint(0, config.vocab_size, (2, 10))
    output = model(tokens)

    assert output.shape == (2, 10, config.embedding_dim)


def test_embedding_batch_independence():
    config = TransformerConfig(dropout=0.0)
    model = TokenAndPositionEmbedding(config)
    model.eval()

    tokens = torch.randint(0, config.vocab_size, (2, 10))
    output = model(tokens)

    assert torch.allclose(output[0], output[1]) is False
