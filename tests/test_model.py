import torch

from emergence.models.transformer.config import TransformerConfig
from emergence.models.transformer.model import TransformerModel


def test_model_output_shape():
    config = TransformerConfig()
    model = TransformerModel(config)

    token_ids = torch.randint(0, config.vocab_size, (2, 10))
    logits = model(token_ids)

    assert logits.shape == (2, 10, config.vocab_size)


def test_model_output_is_finite():
    config = TransformerConfig()
    model = TransformerModel(config)

    token_ids = torch.randint(0, config.vocab_size, (2, 10))
    logits = model(token_ids)

    assert torch.isfinite(logits).all()


def test_model_has_trainable_parameters():
    config = TransformerConfig()
    model = TransformerModel(config)

    assert sum(p.numel() for p in model.parameters()) > 0
    assert all(p.requires_grad for p in model.parameters())
