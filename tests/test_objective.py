"""Next-token alignment, loss masking and padding, checked against values worked out by hand."""

import math

import pytest
import torch
import torch.nn.functional as F

from tiny_transformer.model import Transformer
from tiny_transformer.objective import next_token_loss

TOKENS = torch.tensor([[0, 1, 2, 3]])


def confident(predicted: torch.Tensor, vocab: int = 5) -> torch.Tensor:
  """Logits that put almost all probability on `predicted[t]` at each position t."""
  return F.one_hot(predicted, vocab).float() * 100.0


def test_uniform_logits_cost_log_vocab() -> None:
  # Every target has probability 1/5 under uniform logits, so the loss is ln 5.
  loss = next_token_loss(torch.zeros(1, 4, 5), TOKENS)
  assert loss.item() == pytest.approx(math.log(5))


def test_position_t_is_scored_against_token_t_plus_one() -> None:
  right = confident(torch.tensor([[1, 2, 3, 0]]))  # position t predicts token t + 1
  off_by_one = confident(TOKENS)  # position t "predicts" the token it already sees
  assert next_token_loss(right, TOKENS).item() < 1e-6
  assert next_token_loss(off_by_one, TOKENS).item() > 50


def test_mask_selects_targets_by_token_position() -> None:
  # Only token 3 is a target, so only position 2's prediction counts.
  mask = torch.tensor([[False, False, False, True]])
  logits = confident(torch.tensor([[4, 4, 3, 4]]))  # wrong everywhere except position 2
  assert next_token_loss(logits, TOKENS, mask).item() < 1e-6


def test_a_batch_with_no_targets_raises() -> None:
  with pytest.raises(ValueError, match='no targets'):
    next_token_loss(torch.zeros(1, 4, 5), TOKENS, torch.zeros(1, 4, dtype=torch.bool))
  # The first token can never be a target, so masking only it still leaves none.
  with pytest.raises(ValueError, match='no targets'):
    next_token_loss(torch.zeros(1, 4, 5), TOKENS, torch.tensor([[True, False, False, False]]))


def test_right_padding_matches_the_unpadded_sequences(model: Transformer) -> None:
  short = torch.tensor([[1, 2, 3]])
  long = torch.tensor([[4, 5, 6, 7, 8]])
  pad = 0
  batch = torch.tensor([[1, 2, 3, pad, pad], [4, 5, 6, 7, 8]])
  mask = torch.tensor([[True, True, True, False, False], [True] * 5])

  # Token-weighted mean over both sequences: 2 targets from the short, 4 from the long.
  short_loss = next_token_loss(model(short), short).item()
  long_loss = next_token_loss(model(long), long).item()
  expected = (2 * short_loss + 4 * long_loss) / 6
  # The padded batch runs different-shaped matmuls, so allow float32 rounding.
  assert next_token_loss(model(batch), batch, mask).item() == pytest.approx(expected, rel=1e-5)
