"""
Load a trained tiny-transformer run and have it sort a list.

Pass the values to sort, or leave them out to sort a random list:
  just predict default-seed0 --values 5 3 9 1 ...
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from tiny_transformer.generate import generate
from tiny_transformer.hub import load_model
from tiny_transformer.sorting import from_description


@dataclass
class Config:
  run: Path
  """Run directory, e.g. runs/default-seed0."""
  values: tuple[int, ...] = ()
  """The list to sort. Empty draws a random one."""
  seed: int = 0
  """Seeds the random list when no values are given."""


def main(cfg: Config) -> None:
  task, _ = from_description(json.loads((cfg.run / 'metrics.json').read_text())['data'])
  model, vocab = load_model(cfg.run)
  assert vocab is not None, 'sorting runs save their vocabulary'
  if cfg.values:
    if len(cfg.values) != task.length or not all(0 <= v < task.n_values for v in cfg.values):
      raise SystemExit(f'this run sorts exactly {task.length} values, each in [0, {task.n_values})')
    inputs = torch.tensor([cfg.values])
  else:
    inputs = task.sample_inputs(1, torch.Generator().manual_seed(cfg.seed))

  tokens, _ = task.build(inputs)
  out = generate(model, tokens[:, :task.length + 1], max_new_tokens=task.length)[0, task.length + 1:]
  print('input: ', ' '.join(vocab[t] for t in inputs[0].tolist()))
  print('model: ', ' '.join(vocab[t] for t in out.tolist()))
  print('sorted:', ' '.join(vocab[t] for t in tokens[0, task.length + 1:].tolist()))
  print('correct' if torch.equal(out, tokens[0, task.length + 1:]) else 'WRONG')


if __name__ == '__main__':
  main(tyro.cli(Config))
