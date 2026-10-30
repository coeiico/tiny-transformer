"""Saving and loading evaluation weights with their config and vocabulary."""

import json
from pathlib import Path

import pytest
import torch

from tiny_transformer.hub import CONFIG_FILE, load_model, save_model
from tiny_transformer.model import Transformer

VOCAB = [str(i) for i in range(10)] + ['SEP']


def test_round_trip_preserves_predictions_config_and_vocab(model: Transformer, tokens: torch.Tensor, tmp_path: Path) -> None:
  save_model(tmp_path, model, VOCAB)
  loaded, vocab = load_model(tmp_path)
  assert loaded.cfg == model.cfg
  assert vocab == VOCAB
  assert not loaded.training
  # Same weights, same operations, same device: predictions match exactly.
  assert torch.equal(loaded(tokens), model(tokens))


def test_loading_into_a_different_architecture_fails(model: Transformer, tmp_path: Path) -> None:
  save_model(tmp_path, model, VOCAB)
  record = json.loads((tmp_path / CONFIG_FILE).read_text())
  record['transformer']['n_layers'] = 1  # the weights still hold two blocks
  (tmp_path / CONFIG_FILE).write_text(json.dumps(record))
  with pytest.raises(RuntimeError, match='Unexpected key'):
    load_model(tmp_path)


def test_vocab_must_cover_every_token_id(model: Transformer, tmp_path: Path) -> None:
  with pytest.raises(ValueError, match='vocab'):
    save_model(tmp_path, model, VOCAB[:-1])
