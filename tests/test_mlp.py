import torch

from emergence.models.transformer.config import TransformerConfig
from emergence.models.transformer.mlp import MLP


def test_mlp_output_shape():
    config = TransformerConfig()
    model = MLP(config)

    x = torch.randn(2, 10, config.embedding_dim)
    output = model(x)

    assert output.shape == x.shape


def test_mlp_dropout_disabled_in_eval_mode():
    config = TransformerConfig(dropout=0.5)
    model = MLP(config)
    model.eval()

    x = torch.randn(2, 10, config.embedding_dim)

    output1 = model(x)
    output2 = model(x)

    assert torch.allclose(output1, output2)


def test_mlp_has_trainable_parameters():
    config = TransformerConfig()
    model = MLP(config)

    assert sum(p.numel() for p in model.parameters()) > 0

    assert all(p.requires_grad for p in model.parameters())
