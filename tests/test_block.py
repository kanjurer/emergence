import torch

from emergence.models.transformer.config import TransformerConfig
from emergence.models.transformer.block import TransformerBlock


def test_block_output_shape():
    config = TransformerConfig()
    model = TransformerBlock(config)

    x = torch.randn(2, 10, config.embedding_dim)
    output = model(x)

    assert output.shape == x.shape


def test_block_forward_pass():
    config = TransformerConfig()
    model = TransformerBlock(config)

    x = torch.randn(2, 10, config.embedding_dim)
    output = model(x)

    assert torch.isfinite(output).all()


def test_block_has_trainable_parameters():
    config = TransformerConfig()
    model = TransformerBlock(config)

    assert sum(p.numel() for p in model.parameters()) > 0
    assert all(p.requires_grad for p in model.parameters())
