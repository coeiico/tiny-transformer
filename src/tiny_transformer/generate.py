"""Greedy and temperature sampling, with an optional legal-token mask."""

from collections.abc import Callable

import torch
from jaxtyping import Bool, Int
from torch import Tensor

from tiny_transformer.model import Transformer
from tiny_transformer.modes import evaluating

# Given every token so far, which tokens may come next: True means allowed.
LegalTokens = Callable[[Int[Tensor, 'batch seq']], Bool[Tensor, 'batch vocab']]


def generate(
  model: Transformer,
  prompt: Int[Tensor, 'batch prompt'],
  max_new_tokens: int,
  *,
  temperature: float = 0.0,
  legal: LegalTokens | None = None,
  generator: torch.Generator | None = None,
) -> Int[Tensor, 'batch total']:
  """
  Extend each prompt by `max_new_tokens` tokens, one at a time.

  `temperature=0` picks the most likely token (greedy). Above 0, it samples from
  softmax(logits / temperature): lower is closer to greedy, higher is more random. Pass a
  `generator` on the model's device to make sampling repeatable.

  `legal` is the domain's rule for what may come next, such as legal moves or valid maze
  steps. Illegal tokens get zero probability, and a row with no legal token raises
  ValueError rather than emitting one anyway.

  The whole sequence must fit in `max_seq_len`; there is no silent cropping of the
  context. Runs in eval mode without gradients and restores the model's modes afterwards.
  """
  if temperature < 0:
    raise ValueError(f'temperature must be non-negative, got {temperature}')
  total = prompt.shape[1] + max_new_tokens
  if total > model.cfg.max_seq_len:
    raise ValueError(f'prompt plus new tokens ({total}) exceeds max_seq_len {model.cfg.max_seq_len}')

  tokens = prompt
  with evaluating(model):
    for _ in range(max_new_tokens):
      logits = model(tokens)[:, -1]  # [batch, vocab]: the prediction for the next token
      if legal is not None:
        allowed = legal(tokens)
        if allowed.shape != logits.shape or allowed.dtype != torch.bool:
          raise ValueError(f'legal must return a bool tensor of shape {tuple(logits.shape)}')
        if not allowed.any(dim=-1).all():
          raise ValueError('legal allows no next token for at least one row')
        logits = logits.masked_fill(~allowed, float('-inf'))
      if temperature == 0:
        next_token = logits.argmax(dim=-1)
      else:
        probs = (logits / temperature).softmax(dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
      tokens = torch.cat([tokens, next_token[:, None]], dim=1)
  return tokens
