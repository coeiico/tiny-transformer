"""
Train tiny-transformer on the sorting demo task.

`just quick` runs this with --quick: a tiny CPU smoke test that must finish in under 5
minutes. `just train` runs the full experiment. `--n-layers 0` trains the bigram baseline.
"""

import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use('Agg')  # write PNGs without needing a display
import matplotlib.pyplot as plt
import torch
import tyro

from tiny_transformer.config import TransformerConfig
from tiny_transformer.hub import count_parameters, save_model
from tiny_transformer.model import Transformer
from tiny_transformer.modes import evaluating
from tiny_transformer.objective import next_token_loss
from tiny_transformer.sorting import (
  SortTask,
  Splits,
  describe,
  exact_match,
  held_out,
  key,
  token_accuracy,
)
from tiny_transformer.utils import (
  code_state,
  environment,
  get_device,
  make_run_dir,
  save_json,
  seed_everything,
)


@dataclass(frozen=True)
class ModelArgs:
  """Architecture flags. The vocabulary and sequence length come from the task."""

  n_layers: int = 4
  d_model: int = 256
  n_heads: int = 8
  d_mlp: int = 1024
  mlp: Literal['gelu', 'relu', 'none'] = 'gelu'
  norm: Literal['layernorm', 'rmsnorm', 'none'] = 'layernorm'
  dropout: float = 0.0


@dataclass(frozen=True)
class OptimArgs:
  steps: int = 4000
  batch_size: int = 256
  lr: float = 1e-3
  """Peak learning rate for AdamW."""
  warmup_steps: int = 100
  """Linear warm-up: Adam's early updates are noisy until its moment estimates settle."""
  min_lr_ratio: float = 0.1
  """After warm-up, the learning rate follows a cosine down to this fraction of the peak."""
  weight_decay: float = 0.1
  """Applied to weight matrices and embeddings only, not to biases or norm gains."""
  grad_clip: float = 1.0
  """Clip the global gradient norm to this, so one bad batch cannot blow up the weights."""


@dataclass
class Config:
  run_name: str = 'default'
  """Outputs go to runs/<run-name>/, which must not exist yet."""
  seed: int = 0
  """Seeds Python, NumPy and PyTorch, and the stream of training lists."""
  quick: bool = False
  """Run the tiny CPU smoke test instead of the full experiment."""
  device: Literal['auto', 'cpu', 'cuda', 'mps'] = 'auto'
  eval_every: int = 250
  """Steps between validation checks."""
  task: SortTask = field(default_factory=SortTask)
  splits: Splits = field(default_factory=Splits)
  model: ModelArgs = field(default_factory=ModelArgs)
  optim: OptimArgs = field(default_factory=OptimArgs)


def quick_config(cfg: Config) -> Config:
  """Shrink everything so the whole pipeline runs on a CPU in well under 5 minutes."""
  return replace(
    cfg,
    device='cpu',
    eval_every=100,
    task=SortTask(n_values=10, length=6),
    splits=Splits(val_size=256, test_size=256),
    model=ModelArgs(n_layers=2, d_model=64, n_heads=4, d_mlp=256),
    optim=replace(cfg.optim, steps=400, batch_size=64, warmup_steps=20),
  )


def make_optimizer(model: Transformer, args: OptimArgs) -> torch.optim.AdamW:
  # Weight decay pulls weights towards zero. That suits matrices and embeddings, but biases
  # and norm gains are offsets and scales the decay would only distort (as in nanoGPT).
  decay = [p for p in model.parameters() if p.dim() >= 2]
  no_decay = [p for p in model.parameters() if p.dim() < 2]
  groups = [{'params': decay, 'weight_decay': args.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]
  return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))


def learning_rate(step: int, args: OptimArgs) -> float:
  """
  Linear warm-up, then cosine decay to min_lr_ratio × the peak (as in nanoGPT).

  Large steps late in training keep the weights bouncing around a minimum; decaying the
  rate lets them settle into it.
  """
  if step <= args.warmup_steps:
    return args.lr * step / args.warmup_steps
  progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
  cosine = 0.5 * (1 + math.cos(math.pi * progress))  # 1 → 0
  return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine)


def plot_curves(history: list[dict[str, float]], path: Path) -> None:
  steps = [h['step'] for h in history]
  fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
  ax_loss.plot(steps, [h['train_loss'] for h in history], label='train')
  ax_loss.plot(steps, [h['val_loss'] for h in history], label='validation')
  ax_loss.set(xlabel='step', ylabel='loss on sorted outputs (nats)', yscale='log')
  ax_loss.legend()
  ax_acc.plot(steps, [h['val_token_accuracy'] for h in history], label='token accuracy')
  ax_acc.plot(steps, [h['val_exact_match'] for h in history], label='lists sorted exactly')
  ax_acc.set(xlabel='step', ylabel='validation fraction', ylim=(0, 1.02))
  ax_acc.legend()
  final = history[-1]
  fig.suptitle(
    f'Sorts {final["val_exact_match"]:.0%} of held-out lists exactly by step {final["step"]} '
    f'(token accuracy {final["val_token_accuracy"]:.1%})'
  )
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def main(cfg: Config) -> None:
  if cfg.quick:
    cfg = quick_config(cfg)
  seed_everything(cfg.seed)
  device = get_device() if cfg.device == 'auto' else torch.device(cfg.device)
  run_dir = make_run_dir(cfg.run_name)
  task, optim = cfg.task, cfg.optim

  model_cfg = TransformerConfig(vocab_size=task.vocab_size, max_seq_len=task.seq_len, **asdict(cfg.model))
  model = Transformer(model_cfg).to(device)
  optimizer = make_optimizer(model, optim)
  print(f'tiny-transformer: {count_parameters(model):,} parameters on {device}, writing to {run_dir}')

  # Validation and test lists are fixed; training lists are drawn fresh every step from
  # their own seeded stream, skipping any list that is held out.
  val_inputs, test_inputs = held_out(task, cfg.splits)
  held_out_keys = {key(r) for r in val_inputs} | {key(r) for r in test_inputs}
  train_stream = torch.Generator().manual_seed(cfg.seed)
  val_tokens, val_mask = (t.to(device) for t in task.build(val_inputs))

  vocab = task.vocab()
  for row in task.build(val_inputs[:2])[0]:
    print('  example:', ' '.join(vocab[t] for t in row.tolist()))

  record = {
    'config': asdict(cfg),
    'model_config': model_cfg.to_dict(),
    'parameters': count_parameters(model),
    'data': describe(task, cfg.splits, cfg.seed),
    'code': code_state(run_dir),
    'environment': environment(device),
    'resumable': False,
    'status': 'running',
  }
  save_json(run_dir / 'metrics.json', record)

  history: list[dict[str, float]] = []
  start = time.time()
  for step in range(1, optim.steps + 1):
    lr = learning_rate(step, optim)
    for group in optimizer.param_groups:
      group['lr'] = lr
    tokens, loss_mask = (t.to(device) for t in task.build(task.sample_inputs(optim.batch_size, train_stream, held_out_keys)))

    logits = model(tokens)
    loss = next_token_loss(logits, tokens, loss_mask)
    if not torch.isfinite(loss):
      record.update(status=f'failed: non-finite loss {loss.item()} at step {step}', history=history)
      save_json(run_dir / 'metrics.json', record)
      raise SystemExit(f'Stopping: loss is {loss.item()} at step {step}.')
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optim.grad_clip)
    optimizer.step()

    if step % cfg.eval_every == 0 or step == optim.steps:
      with evaluating(model):
        val_logits = model(val_tokens)
        val_loss = next_token_loss(val_logits, val_tokens, val_mask).item()
        val_acc = token_accuracy(val_logits, val_tokens, val_mask)
      val_exact = exact_match(model, task, val_inputs)
      history.append({
        'step': step,
        'train_loss': loss.item(),
        'val_loss': val_loss,
        'val_token_accuracy': val_acc,
        'val_exact_match': val_exact,
        'grad_norm': grad_norm.item(),
        'lr': lr,
      })
      print(
        f'step {step:5d}  train {loss.item():.4f}  val {val_loss:.4f}  '
        f'token acc {val_acc:.3f}  exact {val_exact:.3f}  grad norm {grad_norm.item():.2f}'
      )

  save_model(run_dir, model, vocab)
  plot_curves(history, run_dir / 'training.png')
  final = history[-1]
  record.update(
    status='finished',
    seconds=round(time.time() - start, 1),
    history=history,
    final={k: v for k, v in final.items() if k.startswith('val_')},
  )
  save_json(run_dir / 'metrics.json', record)
  print(f'done in {record["seconds"]}s: {run_dir}')


if __name__ == '__main__':
  main(tyro.cli(Config))
