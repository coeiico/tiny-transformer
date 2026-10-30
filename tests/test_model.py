"""
The model's contracts: shapes, causality, and the cache and patch interface.

Equality checks between two forward passes over the same inputs use exact equality: both
run the same operations in the same order on the CPU, so any difference is a real bug.
"""

import pytest
import torch

from tiny_transformer.config import TransformerConfig
from tiny_transformer.model import Transformer


def test_logits_have_one_row_per_position(model: Transformer, tokens: torch.Tensor) -> None:
  assert model(tokens).shape == (3, 10, model.cfg.vocab_size)


def test_default_size_is_about_three_million_parameters() -> None:
  n = sum(p.numel() for p in Transformer(TransformerConfig(vocab_size=100)).parameters())
  assert 3_000_000 < n < 3_500_000


@pytest.mark.parametrize('position', [0, 4, 9])
def test_changing_a_token_never_changes_earlier_logits(model: Transformer, tokens: torch.Tensor, position: int) -> None:
  changed = tokens.clone()
  changed[:, position] = (changed[:, position] + 1) % model.cfg.vocab_size
  before, after = model(tokens), model(changed)
  # Masked keys get exactly zero weight, but the matmuls see different values, so allow
  # float32 rounding rather than demanding bitwise equality.
  torch.testing.assert_close(before[:, :position], after[:, :position])
  assert not torch.allclose(before[:, position:], after[:, position:])


def test_sequences_longer_than_the_position_table_are_rejected(model: Transformer) -> None:
  with pytest.raises(ValueError, match='max_seq_len'):
    model(torch.zeros(1, model.cfg.max_seq_len + 1, dtype=torch.long))


def test_cache_does_not_change_logits(model: Transformer, tokens: torch.Tensor) -> None:
  logits, cache = model(tokens, return_cache=True)
  assert torch.equal(logits, model(tokens))
  assert list(cache) == model.hook_names()


def test_cache_shapes_match_the_documented_table(model: Transformer, tokens: torch.Tensor) -> None:
  _, cache = model(tokens, return_cache=True)
  batch, seq = tokens.shape
  cfg = model.cfg
  d, heads = cfg.d_model, cfg.n_heads
  assert cache['embed'].shape == (batch, seq, d)
  assert cache['pos_embed'].shape == (seq, d)
  assert cache['blocks.1.resid_pre'].shape == (batch, seq, d)
  assert cache['blocks.1.attn.pattern'].shape == (batch, heads, seq, seq)
  assert cache['blocks.1.attn.z'].shape == (batch, seq, heads, cfg.d_head)
  assert cache['blocks.1.mlp.post'].shape == (batch, seq, cfg.d_mlp)
  assert cache['blocks.1.resid_post'].shape == (batch, seq, d)


def test_attention_patterns_are_causal_distributions(model: Transformer, tokens: torch.Tensor) -> None:
  _, cache = model(tokens, return_cache=True)
  pattern = cache['blocks.0.attn.pattern']
  torch.testing.assert_close(pattern.sum(dim=-1), torch.ones(pattern.shape[:-1]))
  future = ~torch.tril(torch.ones(10, 10, dtype=torch.bool))
  assert (pattern[..., future] == 0).all()


def test_attention_only_model_has_no_mlp_hooks() -> None:
  model = Transformer(TransformerConfig(vocab_size=5, n_layers=1, d_model=8, n_heads=2, mlp='none'))
  _, cache = model(torch.zeros(1, 4, dtype=torch.long), return_cache=True)
  assert not any('mlp' in name or 'resid_mid' in name for name in cache)


def test_identity_patches_preserve_logits(model: Transformer, tokens: torch.Tensor) -> None:
  clean, cache = model(tokens, return_cache=True)
  as_functions = {name: (lambda x: x) for name in model.hook_names()}
  as_tensors = {name: cache[name] for name in model.hook_names()}
  assert torch.equal(model(tokens, patch=as_functions), clean)
  assert torch.equal(model(tokens, patch=as_tensors), clean)


def test_a_patch_changes_what_follows_and_shows_in_the_cache(model: Transformer, tokens: torch.Tensor) -> None:
  clean = model(tokens)
  zero_heads = {'blocks.1.attn.z': lambda z: torch.zeros_like(z)}
  patched, cache = model(tokens, return_cache=True, patch=zero_heads)
  assert (cache['blocks.1.attn.z'] == 0).all()
  assert not torch.allclose(patched, clean)


def test_unknown_patch_names_are_rejected(model: Transformer, tokens: torch.Tensor) -> None:
  with pytest.raises(KeyError, match='blocks.9.attn.z'):
    model(tokens, patch={'blocks.9.attn.z': lambda z: z})


def test_wrongly_shaped_patches_are_rejected(model: Transformer, tokens: torch.Tensor) -> None:
  with pytest.raises(ValueError, match='shape'):
    model(tokens, patch={'blocks.0.resid_pre': torch.zeros(3, 10, model.cfg.d_model + 1)})
  with pytest.raises(ValueError, match='shape'):
    model(tokens, patch={'blocks.0.attn.pattern': lambda p: p[:, :1]})


def test_cached_activations_carry_gradients(model: Transformer, tokens: torch.Tensor) -> None:
  logits, cache = model(tokens, return_cache=True)
  (grad,) = torch.autograd.grad(logits[:, -1].sum(), cache['blocks.0.resid_post'])
  assert grad.abs().sum() > 0


@pytest.mark.parametrize(
  'fields',
  [{'vocab_size': 5, 'd_model': 30, 'n_heads': 4}, {'vocab_size': 0}, {'vocab_size': 5, 'dropout': 1.0}],
  ids=['heads-do-not-divide', 'empty-vocab', 'dropout-one'],
)
def test_impossible_configs_are_rejected(fields: dict[str, int | float]) -> None:
  with pytest.raises(ValueError):
    TransformerConfig(**fields)  # pyright: ignore[reportArgumentType]


def test_saved_configs_with_unknown_fields_are_rejected() -> None:
  with pytest.raises(ValueError, match='unknown'):
    TransformerConfig.from_dict({'vocab_size': 5, 'n_experts': 8})
