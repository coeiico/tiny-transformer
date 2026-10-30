"""The evaluating() helper restores every submodule's mode and the gradient switch."""

import pytest
import torch

from tiny_transformer.model import Transformer
from tiny_transformer.modes import evaluating


def test_evaluating_switches_off_dropout_and_gradients(model: Transformer) -> None:
  model.train()
  with evaluating(model):
    assert not any(m.training for m in model.modules())
    assert not torch.is_grad_enabled()
  assert torch.is_grad_enabled()


def test_gradients_can_stay_on_for_attribution(model: Transformer, tokens: torch.Tensor) -> None:
  with evaluating(model, grad=True):
    assert model(tokens).requires_grad


def test_frozen_submodules_stay_in_eval_mode_even_after_an_error(model: Transformer) -> None:
  model.train()
  model.blocks[0].eval()  # deliberately frozen
  with pytest.raises(RuntimeError, match='boom'), evaluating(model):
    raise RuntimeError('boom')
  assert model.training and model.blocks[1].training
  assert not any(m.training for m in model.blocks[0].modules())
