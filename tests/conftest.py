"""Shared fixtures: a tiny model small enough that every test runs in milliseconds on a CPU."""

import pytest
import torch

from tiny_transformer.config import TransformerConfig
from tiny_transformer.model import Transformer

TINY = TransformerConfig(vocab_size=11, n_layers=2, d_model=32, n_heads=4, d_mlp=64, max_seq_len=16)


@pytest.fixture
def model() -> Transformer:
  torch.manual_seed(0)
  return Transformer(TINY).eval()


@pytest.fixture
def tokens() -> torch.Tensor:
  return torch.randint(0, TINY.vocab_size, (3, 10), generator=torch.Generator().manual_seed(1))
