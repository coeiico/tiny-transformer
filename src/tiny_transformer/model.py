"""
A GPT-style decoder-only transformer, with named activations for caching and patching.

The architecture follows GPT-2 as nanoGPT writes it: learned token and position
embeddings, pre-norm blocks of causal self-attention and an MLP, each added back into the
residual stream, then a final norm and an unembedding to logits.

Hook points
-----------
`forward(tokens, return_cache=True)` returns `(logits, cache)`, where `cache` maps each hook
name below to the activation at that point. `forward(tokens, patch={name: value})` replaces
an activation as the forward pass reaches it: `value` is either a tensor of exactly the
activation's shape, or a function from the activation to a tensor of the same shape. The
cache holds activations after any patch, so it shows what the rest of the model saw.

| Name                          | Shape                         | What it is                                  |
| ----------------------------- | ----------------------------- | ------------------------------------------- |
| `embed`                       | `[batch, seq, d_model]`       | token embeddings                            |
| `pos_embed`                   | `[seq, d_model]`              | position embeddings, shared across a batch  |
| `blocks.{i}.resid_pre`        | `[batch, seq, d_model]`       | residual stream entering block i            |
| `blocks.{i}.attn.pattern`     | `[batch, head, query, key]`   | attention weights, after softmax            |
| `blocks.{i}.attn.z`           | `[batch, seq, head, d_head]`  | each head's output before W_O mixes heads   |
| `blocks.{i}.attn_out`         | `[batch, seq, d_model]`       | what attention adds to the residual stream  |
| `blocks.{i}.resid_mid`        | `[batch, seq, d_model]`       | stream between attention and MLP (MLP only) |
| `blocks.{i}.mlp.post`         | `[batch, seq, d_mlp]`         | MLP neurons, after the activation (MLP only)|
| `blocks.{i}.mlp_out`          | `[batch, seq, d_model]`       | what the MLP adds (MLP only)                |
| `blocks.{i}.resid_post`       | `[batch, seq, d_model]`       | residual stream leaving block i             |

With `mlp='none'` a block has no `resid_mid`, `mlp.post` or `mlp_out`. `Transformer.hook_names()`
lists the names for a given config.

Cached activations stay attached to the autograd graph, so gradient-based analysis can use
them. Call `.detach()` (and `.cpu()`) before storing them for later.
"""

import math
from collections.abc import Callable, Mapping

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from tiny_transformer.config import TransformerConfig

# A patch value: a replacement tensor, or a function that edits the activation.
Patch = Tensor | Callable[[Tensor], Tensor]
# Passed down through the blocks: takes a hook name and an activation, returns the
# activation to continue with (patched or not) and records it in the cache if one is open.
Hook = Callable[[str, Tensor], Tensor]


class Attention(nn.Module):
  """Multi-head causal self-attention."""

  causal_mask: Tensor

  def __init__(self, cfg: TransformerConfig, prefix: str) -> None:
    super().__init__()
    self.prefix = prefix
    self.n_heads = cfg.n_heads
    self.d_head = cfg.d_head
    self.W_q = nn.Linear(cfg.d_model, cfg.d_model)
    self.W_k = nn.Linear(cfg.d_model, cfg.d_model)
    self.W_v = nn.Linear(cfg.d_model, cfg.d_model)
    self.W_o = nn.Linear(cfg.d_model, cfg.d_model)
    self.attn_dropout = nn.Dropout(cfg.dropout)
    self.out_dropout = nn.Dropout(cfg.dropout)
    # mask[q, k] is True where query q may attend to key k: itself and everything before.
    # Not persistent, so it stays out of the state dict: it is derived from the config.
    mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
    self.register_buffer('causal_mask', mask, persistent=False)

  def forward(self, x: Float[Tensor, 'batch seq d_model'], hook: Hook) -> Float[Tensor, 'batch seq d_model']:
    seq = x.shape[1]
    q = einops.rearrange(self.W_q(x), 'b s (h d) -> b h s d', h=self.n_heads)
    k = einops.rearrange(self.W_k(x), 'b s (h d) -> b h s d', h=self.n_heads)
    v = einops.rearrange(self.W_v(x), 'b s (h d) -> b h s d', h=self.n_heads)

    # Scaling by sqrt(d_head) keeps the scores' variance near 1 at initialisation, so the
    # softmax starts soft rather than saturated (Attention Is All You Need, section 3.2.1).
    scores = einops.einsum(q, k, 'b h q d, b h k d -> b h q k') / math.sqrt(self.d_head)
    # -inf before the softmax gives future keys exactly zero weight.
    scores = scores.masked_fill(~self.causal_mask[:seq, :seq], float('-inf'))
    pattern = hook(f'{self.prefix}.attn.pattern', scores.softmax(dim=-1))
    pattern = self.attn_dropout(pattern)

    z = einops.einsum(pattern, v, 'b h q k, b h k d -> b q h d')
    z = hook(f'{self.prefix}.attn.z', z)
    out = self.W_o(einops.rearrange(z, 'b s h d -> b s (h d)'))
    return self.out_dropout(out)


class MLP(nn.Module):
  """Up projection, activation, down projection."""

  def __init__(self, cfg: TransformerConfig, prefix: str) -> None:
    super().__init__()
    self.prefix = prefix
    self.up = nn.Linear(cfg.d_model, cfg.d_mlp)
    self.act = nn.GELU() if cfg.mlp == 'gelu' else nn.ReLU()
    self.down = nn.Linear(cfg.d_mlp, cfg.d_model)
    self.dropout = nn.Dropout(cfg.dropout)

  def forward(self, x: Float[Tensor, 'batch seq d_model'], hook: Hook) -> Float[Tensor, 'batch seq d_model']:
    post = hook(f'{self.prefix}.mlp.post', self.act(self.up(x)))
    return self.dropout(self.down(post))


def make_norm(cfg: TransformerConfig) -> nn.Module:
  if cfg.norm == 'layernorm':
    return nn.LayerNorm(cfg.d_model)
  if cfg.norm == 'rmsnorm':
    return nn.RMSNorm(cfg.d_model)
  return nn.Identity()


class Block(nn.Module):
  """
  One pre-norm transformer block.

  Each sublayer reads a normalised copy of the residual stream and adds its output back,
  so the stream itself is never normalised in place. Gradients reach early layers through
  the plain additions, which is what lets deep stacks train.
  """

  def __init__(self, cfg: TransformerConfig, index: int) -> None:
    super().__init__()
    self.prefix = f'blocks.{index}'
    self.ln1 = make_norm(cfg)
    self.attn = Attention(cfg, self.prefix)
    self.has_mlp = cfg.mlp != 'none'
    if self.has_mlp:
      self.ln2 = make_norm(cfg)
      self.mlp = MLP(cfg, self.prefix)

  def forward(self, x: Float[Tensor, 'batch seq d_model'], hook: Hook) -> Float[Tensor, 'batch seq d_model']:
    p = self.prefix
    x = hook(f'{p}.resid_pre', x)
    x = x + hook(f'{p}.attn_out', self.attn(self.ln1(x), hook))
    if self.has_mlp:
      x = hook(f'{p}.resid_mid', x)
      x = x + hook(f'{p}.mlp_out', self.mlp(self.ln2(x), hook))
    return hook(f'{p}.resid_post', x)


class Transformer(nn.Module):
  """A decoder-only transformer over integer tokens. See the module docstring for hooks."""

  def __init__(self, cfg: TransformerConfig) -> None:
    super().__init__()
    self.cfg = cfg
    self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
    self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
    self.embed_dropout = nn.Dropout(cfg.dropout)
    self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
    self.ln_final = make_norm(cfg)
    self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
    self.apply(self._init_weights)
    # GPT-2's residual scaling: each block adds two outputs to the stream, so shrinking the
    # projections that write into it by 1/sqrt(2 · n_layers) keeps the stream's variance
    # from growing with depth at initialisation.
    for block in self.blocks:
      assert isinstance(block, Block)
      residual_writers = [block.attn.W_o] + ([block.mlp.down] if block.has_mlp else [])
      for linear in residual_writers:
        nn.init.normal_(linear.weight, std=0.02 / math.sqrt(2 * cfg.n_layers))

  @staticmethod
  def _init_weights(module: nn.Module) -> None:
    # GPT-2 initialisation: small normal weights (std 0.02) and zero biases.
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, std=0.02)

  def hook_names(self) -> list[str]:
    """Every hook name this model reaches on a forward pass, in the order it reaches them."""
    names = ['embed', 'pos_embed']
    for i in range(self.cfg.n_layers):
      p = f'blocks.{i}'
      names += [f'{p}.resid_pre', f'{p}.attn.pattern', f'{p}.attn.z', f'{p}.attn_out']
      if self.cfg.mlp != 'none':
        names += [f'{p}.resid_mid', f'{p}.mlp.post', f'{p}.mlp_out']
      names.append(f'{p}.resid_post')
    return names

  def forward(
    self,
    tokens: Int[Tensor, 'batch seq'],
    *,
    return_cache: bool = False,
    patch: Mapping[str, Patch] | None = None,
  ) -> Float[Tensor, 'batch seq vocab'] | tuple[Float[Tensor, 'batch seq vocab'], dict[str, Tensor]]:
    """
    Logits for the next token at every position.

    Position t's logits depend only on tokens 0..t. With `return_cache=True`, also return
    the activations named in the module docstring. `patch` replaces named activations.
    """
    seq = tokens.shape[1]
    if seq > self.cfg.max_seq_len:
      raise ValueError(f'sequence length {seq} exceeds max_seq_len {self.cfg.max_seq_len}')
    patch = dict(patch or {})
    unknown = set(patch) - set(self.hook_names())
    if unknown:
      raise KeyError(f'unknown hook names {sorted(unknown)}; valid names: {self.hook_names()}')
    cache: dict[str, Tensor] | None = {} if return_cache else None

    def hook(name: str, x: Tensor) -> Tensor:
      if name in patch:
        x = _apply_patch(name, x, patch[name])
      if cache is not None:
        cache[name] = x
      return x

    x = hook('embed', self.token_embed(tokens))
    x = x + hook('pos_embed', self.pos_embed.weight[:seq])
    x = self.embed_dropout(x)
    for block in self.blocks:
      x = block(x, hook)
    logits = self.unembed(self.ln_final(x))
    return (logits, cache) if cache is not None else logits


def _apply_patch(name: str, x: Tensor, value: Patch) -> Tensor:
  new = value(x) if callable(value) else value
  if not isinstance(new, Tensor):
    raise TypeError(f'patch for {name!r} must give a tensor, got {type(new).__name__}')
  if new.shape != x.shape:
    raise ValueError(f'patch for {name!r} has shape {tuple(new.shape)}, expected {tuple(x.shape)}')
  return new.to(device=x.device, dtype=x.dtype)
