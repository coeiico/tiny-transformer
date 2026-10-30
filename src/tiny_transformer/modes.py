"""Temporarily evaluate a model, then put every submodule back exactly as it was."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def evaluating(model: nn.Module, *, grad: bool = False) -> Iterator[nn.Module]:
  """
  Run a block in eval mode, with gradient tracking off unless `grad=True`.

  Eval mode and gradient tracking are separate switches: eval mode turns off dropout,
  while gradients stay available for analysis that needs derivatives, such as attribution.

  On exit, even after an exception, each submodule gets back its own previous mode.
  Calling `model.train()` instead would also switch on submodules that were deliberately
  left in eval mode, such as a frozen component.
  """
  previous = {module: module.training for module in model.modules()}
  model.eval()
  try:
    with torch.set_grad_enabled(grad):
      yield model
  finally:
    for module, was_training in previous.items():
      module.training = was_training
