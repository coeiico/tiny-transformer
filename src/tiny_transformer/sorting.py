"""
The demo task this repo trains on: sort a list of integers.

This is not part of the backbone. Nothing in config.py, model.py, generate.py or hub.py
imports it, and downstream repos bring their own data. Sorting makes a good test of the
backbone because it needs attention (each output has to find the right input) and has an
exact answer to check against.

A sequence for `length` values drawn from [0, n_values) is

  x_1 ... x_n  SEP  y_1 ... y_n

where y is x sorted ascending. Only y is a target: x is random, so there is nothing to
learn from predicting it.

The split unit is the input list x. Validation and test lists are fixed by their own
seeds, and training never sees a list that appears in either.
"""

from dataclasses import asdict, dataclass
from typing import Any

import torch
from jaxtyping import Bool, Int
from torch import Tensor

from tiny_transformer.generate import generate
from tiny_transformer.model import Transformer

# Bump when the sequence layout or sampling changes, so old runs are not compared as equal.
TASK_VERSION = 1


@dataclass(frozen=True)
class SortTask:
  n_values: int = 64
  """Distinct values a list can hold."""
  length: int = 16
  """Values per list."""

  @property
  def sep(self) -> int:
    return self.n_values

  @property
  def vocab_size(self) -> int:
    return self.n_values + 1

  @property
  def seq_len(self) -> int:
    return 2 * self.length + 1

  def vocab(self) -> list[str]:
    return [str(v) for v in range(self.n_values)] + ['SEP']

  def build(self, inputs: Int[Tensor, 'batch length']) -> tuple[Int[Tensor, 'batch seq'], Bool[Tensor, 'batch seq']]:
    """The full token sequences for some input lists, and the mask marking y as targets."""
    sep = torch.full_like(inputs[:, :1], self.sep)
    tokens = torch.cat([inputs, sep, inputs.sort(dim=1).values], dim=1)
    loss_mask = torch.zeros_like(tokens, dtype=torch.bool)
    loss_mask[:, self.length + 1:] = True
    return tokens, loss_mask

  def sample_inputs(self, n: int, generator: torch.Generator, exclude: set[bytes] | None = None) -> Int[Tensor, 'n length']:
    """Draw n random input lists, redrawing any that appear in `exclude`."""
    kept: list[Tensor] = []
    while sum(len(k) for k in kept) < n:
      draw = torch.randint(0, self.n_values, (n, self.length), generator=generator)
      if exclude:
        draw = draw[[key(row) not in exclude for row in draw]]
      kept.append(draw)
    return torch.cat(kept)[:n]


@dataclass(frozen=True)
class Splits:
  """Held-out lists, fixed by their own seeds so every run is measured on the same data."""

  val_seed: int = 1_000_003
  val_size: int = 1024
  test_seed: int = 2_000_003
  test_size: int = 2048


def key(row: Tensor) -> bytes:
  return row.numpy().tobytes()


def held_out(task: SortTask, splits: Splits) -> tuple[Int[Tensor, 'val length'], Int[Tensor, 'test length']]:
  """The validation and test input lists. Test excludes validation, so they never overlap."""
  val = task.sample_inputs(splits.val_size, torch.Generator().manual_seed(splits.val_seed))
  test = task.sample_inputs(
    splits.test_size, torch.Generator().manual_seed(splits.test_seed), exclude={key(r) for r in val}
  )
  return val, test


def describe(task: SortTask, splits: Splits, train_seed: int) -> dict[str, Any]:
  """The data definition recorded with a run: enough to regenerate every split exactly."""
  return {'task': 'sorting', 'version': TASK_VERSION, **asdict(task), **asdict(splits), 'train_seed': train_seed}


def from_description(data: dict[str, Any]) -> tuple[SortTask, Splits]:
  """Rebuild the task and splits a run recorded with `describe`."""
  if data.get('task') != 'sorting' or data.get('version') != TASK_VERSION:
    raise ValueError(f'expected sorting task version {TASK_VERSION}, got {data.get("task")} version {data.get("version")}')
  task = SortTask(n_values=data['n_values'], length=data['length'])
  splits = Splits(val_seed=data['val_seed'], val_size=data['val_size'], test_seed=data['test_seed'], test_size=data['test_size'])
  return task, splits


def token_accuracy(logits: Tensor, tokens: Tensor, loss_mask: Tensor) -> float:
  """Fraction of target tokens the model ranks first, given the true tokens before each."""
  correct = logits[:, :-1].argmax(dim=-1) == tokens[:, 1:]
  return correct[loss_mask[:, 1:]].float().mean().item()


def exact_match(model: Transformer, task: SortTask, inputs: Tensor, batch_size: int = 512) -> float:
  """
  Fraction of lists the model sorts perfectly, generating its own output greedily.

  Stricter than token accuracy: one wrong token spoils the list, and an early mistake can
  derail everything after it because the model reads its own output.
  """
  device = next(model.parameters()).device
  hits = 0
  for start in range(0, len(inputs), batch_size):
    tokens, _ = task.build(inputs[start:start + batch_size])
    tokens = tokens.to(device)
    prompt = tokens[:, :task.length + 1]
    out = generate(model, prompt, max_new_tokens=task.length)
    hits += (out[:, task.length + 1:] == tokens[:, task.length + 1:]).all(dim=1).sum().item()
  return hits / len(inputs)
