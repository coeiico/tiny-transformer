"""
Baselines, probes, interventions and plots for a saved tiny-transformer run.

Analysis reads saved weights, so it never needs to retrain.
"""

from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Config:
  run: Path
  """Run directory, e.g. runs/default."""


def main(cfg: Config) -> None:
  # TODO: load the run, then run the experiments listed in README.md and save plots to cfg.run.
  raise SystemExit(f'analyze.py is not implemented yet (run: {cfg.run})')


if __name__ == '__main__':
  main(tyro.cli(Config))
