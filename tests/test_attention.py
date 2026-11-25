import torch

from emergence.models.transformer.config import TransformerConfig
from emergence.models.transformer.attention import CausalSelfAttention


def test_attention_output_shape():
    config = TransformerConfig()
    model = CausalSelfAttention(config)

    x = torch.randn(2, 10, config.embedding_dim)
    output = model(x)

    assert output.shape == x.shape


def test_attention_respects_causal_mask():
    config = TransformerConfig(dropout=0.0)
    model = CausalSelfAttention(config)
    model.eval()

    x1 = torch.randn(1, 5, config.embedding_dim)
    x2 = x1.clone()

    x2[0, 4] = torch.randn(config.embedding_dim)

    output1 = model(x1)
    output2 = model(x2)

    assert torch.allclose(output1[:, :4], output2[:, :4], atol=1e-6)
