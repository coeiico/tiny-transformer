"""Load a trained tiny-transformer run and run inference on it."""

from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Config:
  run: Path
  """Run directory, e.g. runs/default."""


def main(cfg: Config) -> None:
  # TODO: load the weights from cfg.run and run inference.
  raise SystemExit(f'predict.py is not implemented yet (run: {cfg.run})')


if __name__ == '__main__':
  main(tyro.cli(Config))
