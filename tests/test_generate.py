"""Greedy and sampled generation, the legal-token mask, and mode restoration."""

import pytest
import torch

from tiny_transformer.generate import generate
from tiny_transformer.model import Transformer


def test_greedy_generation_appends_the_most_likely_token(model: Transformer) -> None:
  prompt = torch.tensor([[1, 2, 3]])
  out = generate(model, prompt, max_new_tokens=1)
  assert torch.equal(out[:, :3], prompt)
  assert out[0, 3] == model(prompt)[0, -1].argmax()


def test_generation_extends_by_exactly_the_requested_tokens(model: Transformer) -> None:
  out = generate(model, torch.zeros(2, 4, dtype=torch.long), max_new_tokens=5)
  assert out.shape == (2, 9)


def test_illegal_tokens_are_never_generated(model: Transformer) -> None:
  def only_seven(tokens: torch.Tensor) -> torch.Tensor:
    allowed = torch.zeros(tokens.shape[0], model.cfg.vocab_size, dtype=torch.bool)
    allowed[:, 7] = True
    return allowed

  out = generate(model, torch.zeros(2, 2, dtype=torch.long), max_new_tokens=6, temperature=1.0, legal=only_seven)
  assert (out[:, 2:] == 7).all()


def test_a_row_with_no_legal_token_raises(model: Transformer) -> None:
  def nothing(tokens: torch.Tensor) -> torch.Tensor:
    return torch.zeros(tokens.shape[0], model.cfg.vocab_size, dtype=torch.bool)

  with pytest.raises(ValueError, match='no next token'):
    generate(model, torch.zeros(1, 2, dtype=torch.long), max_new_tokens=1, legal=nothing)


def test_sampling_repeats_with_the_same_generator(model: Transformer) -> None:
  prompt = torch.zeros(4, 2, dtype=torch.long)
  a = generate(model, prompt, 8, temperature=1.0, generator=torch.Generator().manual_seed(3))
  b = generate(model, prompt, 8, temperature=1.0, generator=torch.Generator().manual_seed(3))
  assert torch.equal(a, b)


def test_generation_past_max_seq_len_raises(model: Transformer) -> None:
  with pytest.raises(ValueError, match='max_seq_len'):
    generate(model, torch.zeros(1, 10, dtype=torch.long), max_new_tokens=model.cfg.max_seq_len)


def test_generation_restores_training_mode(model: Transformer) -> None:
  model.train()
  generate(model, torch.zeros(1, 2, dtype=torch.long), max_new_tokens=2)
  assert all(m.training for m in model.modules())
