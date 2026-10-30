"""
Save and load evaluation weights with their config and vocabulary, and count parameters.

A saved model is a folder in the Hugging Face Hub's layout: `model.safetensors` holds the
weights and `model_config.json` the architecture and vocabulary, so loading never rebuilds
a model from CLI defaults. `just push <run>` uploads a run folder to the Hub.

These are evaluation weights only. A resumable checkpoint would also need the optimizer
state, step count and random state; nothing here saves those.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from tiny_transformer.config import TransformerConfig
from tiny_transformer.model import Transformer

WEIGHTS_FILE = 'model.safetensors'
CONFIG_FILE = 'model_config.json'
FORMAT_VERSION = 1


def save_model(directory: Path, model: Transformer, vocab: Sequence[str] | None = None) -> None:
  """Write the weights, the architecture config and (if given) each token id's meaning."""
  if vocab is not None and len(vocab) != model.cfg.vocab_size:
    raise ValueError(f'vocab has {len(vocab)} entries but vocab_size is {model.cfg.vocab_size}')
  directory.mkdir(parents=True, exist_ok=True)
  state = {name: t.detach().cpu().contiguous() for name, t in model.state_dict().items()}
  save_file(state, directory / WEIGHTS_FILE)
  record = {
    'format_version': FORMAT_VERSION,
    'transformer': model.cfg.to_dict(),
    'vocab': list(vocab) if vocab is not None else None,
  }
  (directory / CONFIG_FILE).write_text(json.dumps(record, indent=2) + '\n')


def load_model(directory: Path, device: torch.device | str = 'cpu') -> tuple[Transformer, list[str] | None]:
  """
  Rebuild a saved model in eval mode, with its vocabulary.

  Loading is strict: a missing, unexpected or wrongly shaped weight is an error.
  """
  record = json.loads((directory / CONFIG_FILE).read_text())
  if record.get('format_version') != FORMAT_VERSION:
    raise ValueError(f'{directory / CONFIG_FILE} has format {record.get("format_version")}, expected {FORMAT_VERSION}')
  model = Transformer(TransformerConfig.from_dict(record['transformer']))
  model.load_state_dict(load_file(directory / WEIGHTS_FILE), strict=True)
  return model.to(device).eval(), record['vocab']


def count_parameters(model: nn.Module) -> int:
  return sum(p.numel() for p in model.parameters())
