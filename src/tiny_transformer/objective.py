"""The next-token objective: what the backbone is trained to do, whatever the tokens mean."""

import einops
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor


def next_token_loss(
  logits: Float[Tensor, 'batch seq vocab'],
  tokens: Int[Tensor, 'batch seq'],
  loss_mask: Bool[Tensor, 'batch seq'] | None = None,
) -> Float[Tensor, '']:
  """
  Mean cross-entropy of predicting each token from the tokens before it.

  `logits[:, t]` is the prediction for `tokens[:, t + 1]`, so the last position's logits
  have no target and the first token is never a target. `loss_mask[b, t]` says whether
  token t counts as a target: mark only the part of a sequence worth learning (a maze's
  solution, a game's moves), and mark padding False. Right padding needs nothing more,
  because causal attention means no real position ever sees the pads that follow it.
  `loss_mask[:, 0]` is ignored.

  The mean is over every counted target in the batch, so each target weighs the same
  whichever sequence it sits in. A batch with no counted targets raises ValueError: a loss
  of zero there would make a broken mask look like a perfect model.
  """
  predictions = einops.rearrange(logits[:, :-1], 'b s v -> (b s) v')
  targets = tokens[:, 1:]
  per_target = F.cross_entropy(predictions, targets.flatten(), reduction='none').view_as(targets)
  if loss_mask is None:
    return per_target.mean()
  counted = loss_mask[:, 1:]
  if not counted.any():
    raise ValueError('loss_mask leaves no targets in this batch')
  return per_target[counted].mean()
