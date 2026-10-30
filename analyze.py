"""
Baselines, head ablations and attention analysis for a saved tiny-transformer run.

Analysis reads saved weights, so it never needs to retrain. Results go to
runs/<run>/analysis/<name>/, a new folder each time, so the run's own record is never
changed.

- Test metrics for the trained model and for an untrained copy of the same architecture.
- Mean ablation of every attention head: replace one head's output with its average
  (taken from the validation set, not the test set) and measure how much the test loss
  rises. An identity patch is the control: it must change nothing.
- Two lookup scores for every head: how often its strongest attention from an output
  position lands on an input holding the value the model writes next, and on one holding
  the value it has just written. Each comes with its chance level.
"""

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import tyro
from torch import Tensor

from tiny_transformer.hub import load_model
from tiny_transformer.model import Transformer
from tiny_transformer.modes import evaluating
from tiny_transformer.objective import next_token_loss
from tiny_transformer.sorting import (
  SortTask,
  exact_match,
  from_description,
  held_out,
  token_accuracy,
)
from tiny_transformer.utils import (
  code_state,
  environment,
  get_device,
  make_output_dir,
  save_json,
  seed_everything,
)


@dataclass
class Config:
  run: Path
  """Run directory, e.g. runs/default-seed0."""
  name: str | None = None
  """Output folder under <run>/analysis/. Defaults to the current time."""
  device: Literal['auto', 'cpu', 'cuda', 'mps'] = 'auto'


def evaluate(model: Transformer, task: SortTask, inputs: Tensor) -> dict[str, float]:
  device = next(model.parameters()).device
  tokens, mask = (t.to(device) for t in task.build(inputs))
  with evaluating(model):
    logits = model(tokens)
    return {
      'loss': next_token_loss(logits, tokens, mask).item(),
      'token_accuracy': token_accuracy(logits, tokens, mask),
      'exact_match': exact_match(model, task, inputs),
    }


def head_ablations(model: Transformer, task: SortTask, val_inputs: Tensor, test_inputs: Tensor) -> dict[str, Any]:
  """
  Test loss with each head mean-ablated, then each layer's heads all at once.

  Ablating one head at a time misses heads that back each other up: if several heads do
  the same job, removing any one changes little. Ablating a whole layer's attention
  catches that. The identity patch is the control and must change nothing.
  """
  device = next(model.parameters()).device
  val_tokens, _ = (t.to(device) for t in task.build(val_inputs))
  tokens, mask = (t.to(device) for t in task.build(test_inputs))
  n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
  with evaluating(model):
    clean_logits = model(tokens)
    clean_loss = next_token_loss(clean_logits, tokens, mask).item()

    identity = {f'blocks.{layer}.attn.z': (lambda z: z) for layer in range(n_layers)}
    identity_logits = model(tokens, patch=identity)

    # Each head's average output over validation lists and positions: what the rest of
    # the model gets when that head stops carrying information about this particular list.
    _, val_cache = model(val_tokens, return_cache=True)
    clean_accuracy = token_accuracy(clean_logits, tokens, mask)

    def ablated(layer: int, heads: list[int]) -> tuple[float, float]:
      """Loss rise and token-accuracy drop with these heads replaced by their means."""
      name = f'blocks.{layer}.attn.z'
      mean_z = val_cache[name].mean(dim=(0, 1))  # [head, d_head]

      def ablate(z: Tensor) -> Tensor:
        z = z.clone()
        z[:, :, heads] = mean_z[heads]
        return z

      logits = model(tokens, patch={name: ablate})
      return (
        next_token_loss(logits, tokens, mask).item() - clean_loss,
        clean_accuracy - token_accuracy(logits, tokens, mask),
      )

    head_effects = [[ablated(layer, [head]) for head in range(n_heads)] for layer in range(n_layers)]
    layer_effects = [ablated(layer, list(range(n_heads))) for layer in range(n_layers)]
  return {
    'clean_loss': clean_loss,
    'identity_patch_max_logit_change': (identity_logits - clean_logits).abs().max().item(),
    'loss_rise': [[rise for rise, _ in row] for row in head_effects],
    'token_accuracy_drop': [[drop for _, drop in row] for row in head_effects],
    'layer_loss_rise': [rise for rise, _ in layer_effects],
    'layer_token_accuracy_drop': [drop for _, drop in layer_effects],
  }


def lookup_scores(model: Transformer, task: SortTask, inputs: Tensor, offset: int) -> tuple[Tensor, float, dict[str, Tensor]]:
  """
  How often each head's strongest attention to the input lands on a particular value.

  `offset=1` asks about the value about to be written: queries run from SEP to the
  second-last output, and query t's value of interest is token t + 1. `offset=0` asks about
  the value just written: queries are the outputs themselves. A query's score is 1 when its
  largest attention weight over the input falls on a copy of that value.

  Also returns chance, the score from picking an input position at random: the fraction
  of the input holding the value of interest.
  """
  device = next(model.parameters()).device
  tokens, _ = (t.to(device) for t in task.build(inputs))
  n = task.length
  queries = torch.arange(n + 1 - offset, 2 * n + 1 - offset, device=device)
  targets = tokens[:, queries + offset]  # [batch, n]
  holds_target = tokens[:, None, :n] == targets[:, :, None]  # [batch, query, input key]
  chance = holds_target.float().mean().item()
  with evaluating(model):
    _, cache = model(tokens, return_cache=True)
  scores = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
  for layer in range(model.cfg.n_layers):
    pattern = cache[f'blocks.{layer}.attn.pattern'][:, :, queries, :n]  # [batch, head, query, input key]
    looked_at = pattern.argmax(dim=-1)  # [batch, head, query]
    value_seen = torch.gather(tokens[:, None, :n].expand(-1, pattern.shape[1], -1), 2, looked_at)
    scores[layer] = (value_seen == targets[:, None, :]).float().mean(dim=(0, 2)).cpu()
  return scores, chance, cache


def heatmap(values: Tensor, title: str, label: str, path: Path) -> None:
  fig, ax = plt.subplots(figsize=(1.2 + 0.7 * values.shape[1], 1.4 + 0.6 * values.shape[0]))
  image = ax.imshow(values.numpy(), cmap='viridis', aspect='auto')
  for layer in range(values.shape[0]):
    for head in range(values.shape[1]):
      ax.text(head, layer, f'{values[layer, head]:.2f}', ha='center', va='center', color='white', fontsize=8)
  ax.set(xlabel='head', ylabel='layer', xticks=range(values.shape[1]), yticks=range(values.shape[0]))
  fig.colorbar(image, ax=ax, label=label)
  ax.set_title(title, fontsize=10)
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def plot_pattern(cache: dict[str, Tensor], tokens: Tensor, vocab: list[str], layer: int, head: int, title: str, path: Path) -> None:
  pattern = cache[f'blocks.{layer}.attn.pattern'][0, head].detach().cpu()
  labels = [vocab[t] for t in tokens[0].tolist()]
  fig, ax = plt.subplots(figsize=(8, 7))
  ax.imshow(pattern.numpy(), cmap='Blues')
  ax.set_xticks(range(len(labels)), labels, fontsize=7)
  ax.set_yticks(range(len(labels)), labels, fontsize=7)
  ax.set(xlabel='key (attended to)', ylabel='query (attending)')
  ax.set_title(title, fontsize=10)
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def main(cfg: Config) -> None:
  record = json.loads((cfg.run / 'metrics.json').read_text())
  task, splits = from_description(record['data'])
  device = get_device() if cfg.device == 'auto' else torch.device(cfg.device)
  out = make_output_dir(cfg.run / 'analysis' / (cfg.name or time.strftime('%Y%m%d-%H%M%S')))
  model, vocab = load_model(cfg.run, device)
  assert vocab is not None, 'sorting runs save their vocabulary'
  val_inputs, test_inputs = held_out(task, splits)

  # The untrained baseline uses the run's own seed, so it is the model training started from.
  seed_everything(record['config']['seed'])
  untrained = Transformer(model.cfg).to(device)
  results: dict[str, Any] = {
    'run': str(cfg.run),
    'code': code_state(out),
    'environment': environment(device),
    'test_size': len(test_inputs),
    'chance_token_accuracy': 1 / task.n_values,
    'trained': evaluate(model, task, test_inputs),
    'untrained': evaluate(untrained, task, test_inputs),
  }
  print(f'trained   {results["trained"]}')
  print(f'untrained {results["untrained"]}')

  # The numbers behind the README figures (figure-design.md): copy a file into figures/data/
  # to show it there. Named after the run, so several runs can sit side by side.
  fig_dir = out / 'figure-data'
  fig_dir.mkdir()
  tag = cfg.run.name
  fields = ['step', 'train_loss', 'val_loss', 'val_token_accuracy', 'val_exact_match']
  with (fig_dir / f'curves-{tag}.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for h in record['history']:
      writer.writerow({k: h[k] for k in fields})

  if model.cfg.n_layers > 0:
    ablations = head_ablations(model, task, val_inputs, test_inputs)
    scores, chance, cache = lookup_scores(model, task, test_inputs, offset=1)
    current, current_chance, _ = lookup_scores(model, task, test_inputs, offset=0)
    results['head_ablation'] = ablations
    results['lookup_next'] = {'score': scores.tolist(), 'chance': chance}
    results['lookup_current'] = {'score': current.tolist(), 'chance': current_chance}
    print(f'identity patch changes logits by at most {ablations["identity_patch_max_logit_change"]:.2e}')

    for layer, (rise, drop) in enumerate(zip(ablations['layer_loss_rise'], ablations['layer_token_accuracy_drop'], strict=True)):
      print(f'ablating all of layer {layer} attention: loss +{rise:.2f} nats, token accuracy -{drop:.1%}')

    loss_rise = torch.tensor(ablations['loss_rise'])
    worst = divmod(int(loss_rise.argmax()), model.cfg.n_heads)
    heatmap(
      loss_rise,
      f'Ablating head {worst[0]}.{worst[1]} raises test loss most (+{loss_rise.max():.2f} nats)',
      'test loss rise (nats)',
      out / 'head_ablation.png',
    )
    best = divmod(int(scores.argmax()), model.cfg.n_heads)
    finding = (
      f'Head {best[0]}.{best[1]} attends most to the next value to write on '
      f'{scores.max():.0%} of outputs (chance {chance:.0%})'
    )
    heatmap(scores, finding, 'next-value lookup score', out / 'lookup_next.png')
    heatmap(
      current,
      f'Head {worst[0]}.{worst[1]} attends most to the value just written on '
      f'{current[worst]:.0%} of outputs (chance {current_chance:.0%})',
      'current-value lookup score',
      out / 'lookup_current.png',
    )
    tokens, _ = task.build(test_inputs[:1])

    with (fig_dir / f'heads-{tag}.csv').open('w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(['layer', 'head', 'loss_rise', 'token_accuracy_drop', 'lookup_current', 'lookup_next'])
      for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
          writer.writerow([
            layer, head, ablations['loss_rise'][layer][head], ablations['token_accuracy_drop'][layer][head],
            current[layer, head].item(), scores[layer, head].item(),
          ])
    with (fig_dir / f'layers-{tag}.csv').open('w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(['layer', 'loss_rise', 'token_accuracy_drop'])
      for layer in range(model.cfg.n_layers):
        writer.writerow([layer, ablations['layer_loss_rise'][layer], ablations['layer_token_accuracy_drop'][layer]])
    # The critical head's attention on the first test list, with the keys it is expected to
    # attend to: for each output query, the input positions holding the value just written.
    n = task.length
    row = tokens[0].tolist()
    expected = [[q, k] for q in range(n + 1, 2 * n + 1) for k in range(n) if row[k] == row[q]]
    save_json(fig_dir / f'pattern-{tag}.json', {
      'layer': worst[0],
      'head': worst[1],
      'tokens': [vocab[t] for t in row],
      'input_length': n,
      'pattern': cache[f'blocks.{worst[0]}.attn.pattern'][0, worst[1]].detach().cpu().tolist(),
      'expected': expected,
    })

    plot_pattern(cache, tokens, vocab, best[0], best[1], finding, out / 'lookup_next_pattern.png')
    plot_pattern(
      cache, tokens, vocab, worst[0], worst[1],
      f'Head {worst[0]}.{worst[1]} (+{loss_rise.max():.2f} nats when ablated) attends from each output '
      f'to the input copy of the value just written ({current[worst]:.0%} of outputs)',
      out / 'critical_pattern.png',
    )

  save_json(out / 'results.json', results)
  print(f'wrote {out}')


if __name__ == '__main__':
  main(tyro.cli(Config))
