import pytest

from emergence.models.transformer.config import TransformerConfig


def test_default_config():
    config = TransformerConfig()

    assert config.vocab_size == 256
    assert config.embedding_dim == 128
    assert config.num_layers == 4
    assert config.num_heads == 4


def test_invalid_embedding_dimensions():
    with pytest.raises(ValueError):
        TransformerConfig(embedding_dim=127, num_heads=4)


def test_invalid_dropout():
    with pytest.raises(ValueError):
        TransformerConfig(dropout=1.0)
