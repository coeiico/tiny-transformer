"""Learning checks: gradients reach every parameter, and a tiny model can memorise a tiny batch."""

import pytest
import torch

from tiny_transformer.config import TransformerConfig
from tiny_transformer.model import Transformer
from tiny_transformer.objective import next_token_loss


@pytest.mark.parametrize('mlp', ['gelu', 'none'])
@pytest.mark.parametrize('norm', ['layernorm', 'rmsnorm', 'none'])
def test_every_parameter_receives_a_gradient(mlp: str, norm: str) -> None:
  torch.manual_seed(0)
  cfg = TransformerConfig(vocab_size=7, n_layers=2, d_model=16, n_heads=2, d_mlp=32, max_seq_len=12, mlp=mlp, norm=norm)  # pyright: ignore[reportArgumentType]
  model = Transformer(cfg)
  tokens = torch.randint(0, 7, (4, 8), generator=torch.Generator().manual_seed(0))
  next_token_loss(model(tokens), tokens).backward()
  for name, p in model.named_parameters():
    assert p.grad is not None, f'{name} got no gradient'
    assert torch.isfinite(p.grad).all(), f'{name} has a non-finite gradient'
    if name == 'pos_embed.weight':
      # Positions 0-6 feed predictions. Position 7 is the last: its logits have no target and
      # causal attention means nothing else reads it, so it gets no gradient, and neither
      # do positions the batch never reaches.
      assert (p.grad[:7].abs().sum(dim=1) > 0).all()
      assert p.grad[7:].abs().sum() == 0
    else:
      assert p.grad.abs().sum() > 0, f'{name} got an all-zero gradient'


def test_bigram_model_trains_its_embeddings() -> None:
  model = Transformer(TransformerConfig(vocab_size=7, n_layers=0, d_model=16, max_seq_len=8))
  tokens = torch.randint(0, 7, (4, 8), generator=torch.Generator().manual_seed(0))
  next_token_loss(model(tokens), tokens).backward()
  assert model.token_embed.weight.grad is not None and model.token_embed.weight.grad.abs().sum() > 0


def test_tiny_model_overfits_a_tiny_batch() -> None:
  torch.manual_seed(0)
  model = Transformer(TransformerConfig(vocab_size=11, n_layers=2, d_model=32, n_heads=4, d_mlp=64, max_seq_len=16))
  tokens = torch.randint(0, 11, (4, 12), generator=torch.Generator().manual_seed(0))
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
  losses = []
  for _ in range(300):
    loss = next_token_loss(model(tokens), tokens)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  # Random tokens are unpredictable in general (ln 11 ≈ 2.4 nats), so a near-zero loss
  # means the model memorised these four sequences.
  assert losses[0] > 2.0
  assert losses[-1] < 0.05
