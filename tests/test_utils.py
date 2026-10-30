"""Run bookkeeping: devices, seeding and run folders."""

from pathlib import Path

import pytest
import torch

from tiny_transformer.utils import get_device, make_run_dir, seed_everything


def test_device_is_detected() -> None:
  assert isinstance(get_device(), torch.device)


def test_seeding_repeats_exactly() -> None:
  seed_everything(0)
  a = torch.rand(3)
  seed_everything(0)
  b = torch.rand(3)
  assert torch.equal(a, b)


def test_existing_run_folders_are_refused(tmp_path: Path) -> None:
  make_run_dir('a', root=tmp_path)
  with pytest.raises(FileExistsError):
    make_run_dir('a', root=tmp_path)
