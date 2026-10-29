"""
Train tiny-transformer.

`just quick` runs this with --quick: a tiny CPU smoke test that must finish in under 5
minutes. `just train` runs the full experiment.
"""

from dataclasses import dataclass

import torch
import tyro

from tiny_transformer.utils import get_device, make_run_dir, save_metrics, seed_everything


@dataclass
class Config:
  run_name: str = 'default'
  """Outputs go to runs/<run-name>/."""
  seed: int = 0
  """Seeds Python, NumPy and PyTorch."""
  quick: bool = False
  """Run the tiny CPU smoke test instead of the full experiment."""


def main(cfg: Config) -> None:
  seed_everything(cfg.seed)
  # The smoke test should behave the same on every machine, so it stays on the CPU.
  device = torch.device('cpu') if cfg.quick else get_device()
  run_dir = make_run_dir(cfg.run_name)
  print(f'tiny-transformer: device={device}, run_dir={run_dir}')

  # TODO: build the data, model and optimizer, then write the training loop with
  # the four core lines visible: forward, zero_grad, backward, step. Save weights
  # as run_dir / 'model.safetensors' and plots as PNGs in run_dir.

  save_metrics(run_dir, cfg, {'status': 'scaffold: training not implemented yet'})


if __name__ == '__main__':
  main(tyro.cli(Config))
