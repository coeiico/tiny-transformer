"""The sorting demo task: sequence layout, targets, determinism and split separation."""

import pytest
import torch

from tiny_transformer.sorting import (
  SortTask,
  Splits,
  describe,
  from_description,
  held_out,
  key,
  token_accuracy,
)

TASK = SortTask(n_values=10, length=6)


def test_sequences_are_input_sep_then_sorted_input() -> None:
  inputs = torch.tensor([[3, 1, 2, 9, 1, 0]])
  tokens, mask = TASK.build(inputs)
  assert tokens.tolist() == [[3, 1, 2, 9, 1, 0, TASK.sep, 0, 1, 1, 2, 3, 9]]
  assert mask.tolist() == [[False] * 7 + [True] * 6]


def test_sampling_is_reproducible_and_in_range() -> None:
  a = TASK.sample_inputs(50, torch.Generator().manual_seed(0))
  b = TASK.sample_inputs(50, torch.Generator().manual_seed(0))
  assert torch.equal(a, b)
  assert a.min() >= 0 and a.max() < TASK.n_values


def test_excluded_lists_are_never_sampled() -> None:
  # A tiny space (2 values, length 3: 8 lists) makes collisions certain without exclusion.
  task = SortTask(n_values=2, length=3)
  excluded = task.sample_inputs(4, torch.Generator().manual_seed(0))
  keys = {key(r) for r in excluded}
  drawn = task.sample_inputs(200, torch.Generator().manual_seed(1), exclude=keys)
  assert len(drawn) == 200
  assert not any(key(r) in keys for r in drawn)


def test_validation_and_test_never_overlap() -> None:
  val, test = held_out(TASK, Splits(val_size=300, test_size=300))
  assert not {key(r) for r in val} & {key(r) for r in test}


def test_description_round_trips() -> None:
  splits = Splits(val_size=10, test_size=20)
  assert from_description(describe(TASK, splits, train_seed=0)) == (TASK, splits)


def test_old_task_versions_are_rejected() -> None:
  with pytest.raises(ValueError, match='version'):
    from_description({**describe(TASK, Splits(), 0), 'version': 0})


def test_token_accuracy_counts_only_targets() -> None:
  tokens, mask = TASK.build(torch.tensor([[3, 1, 2, 9, 1, 0]]))
  # Right on every output target, wrong on every input position.
  predicted = torch.cat([tokens[:, 1:], tokens[:, :1]], dim=1)
  predicted[:, :6] = (predicted[:, :6] + 1) % TASK.vocab_size
  logits = torch.nn.functional.one_hot(predicted, TASK.vocab_size).float()
  assert token_accuracy(logits, tokens, mask) == 1.0
