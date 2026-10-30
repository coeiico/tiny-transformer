"""The TransformerConfig dataclass: everything needed to rebuild a model from its weights."""

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class TransformerConfig:
  """
  Architecture of a decoder-only transformer.

  Frozen, because a saved run rebuilds its model from this record alone: changing it after
  the model exists would make the record lie about the weights.
  """

  vocab_size: int
  """Number of distinct token ids. Tokens must lie in [0, vocab_size)."""
  n_layers: int = 4
  """Transformer blocks. 0 gives a bigram model: embed, then unembed."""
  d_model: int = 256
  """Width of the residual stream."""
  n_heads: int = 8
  """Attention heads per block. Each head has width d_model // n_heads."""
  d_mlp: int = 1024
  """Hidden width of each MLP, conventionally 4 × d_model."""
  max_seq_len: int = 256
  """Longest sequence the learned position embeddings cover."""
  mlp: Literal['gelu', 'relu', 'none'] = 'gelu'
  """MLP activation; 'none' removes the MLPs for an attention-only model."""
  norm: Literal['layernorm', 'rmsnorm', 'none'] = 'layernorm'
  """Normalisation before each sublayer and before the unembedding."""
  dropout: float = 0.0
  """Dropout on attention patterns and sublayer outputs. 0 keeps activations deterministic."""

  def __post_init__(self) -> None:
    # Catch impossible shapes here, with a clear message, rather than as a reshape error deep
    # inside the attention code.
    if self.vocab_size < 1:
      raise ValueError(f'vocab_size must be positive, got {self.vocab_size}')
    if self.n_layers < 0:
      raise ValueError(f'n_layers must be non-negative, got {self.n_layers}')
    if self.d_model < 1 or self.max_seq_len < 1:
      raise ValueError('d_model and max_seq_len must be positive')
    if self.n_layers > 0 and (self.n_heads < 1 or self.d_model % self.n_heads != 0):
      raise ValueError(f'd_model ({self.d_model}) must divide evenly into n_heads ({self.n_heads})')
    if self.mlp != 'none' and self.d_mlp < 1:
      raise ValueError(f'd_mlp must be positive when mlp is {self.mlp!r}')
    if not 0.0 <= self.dropout < 1.0:
      raise ValueError(f'dropout must be in [0, 1), got {self.dropout}')

  @property
  def d_head(self) -> int:
    return self.d_model // self.n_heads

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, record: dict[str, Any]) -> 'TransformerConfig':
    """Rebuild a config from a saved record, rejecting fields this version does not know."""
    unknown = set(record) - set(cls.__dataclass_fields__)
    if unknown:
      raise ValueError(f'unknown TransformerConfig fields in saved record: {sorted(unknown)}')
    return cls(**record)
