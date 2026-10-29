# tiny-transformer

**The shared GPT-style backbone.** A readable, hookable GPT-style decoder-only transformer that the other Latent Space Project repos install and build on.

Part of the **Latent Space Project**: 18 small PyTorch repos that test one idea, that neural networks encode concepts and strategy in their latent space, and that we can read and edit those representations. Everything trains on a single consumer GPU or a laptop.

| Roadmap | Phase | Depends on | Status |
| --- | --- | --- | --- |
| 3 of 18 | 1. Foundations | Lessons from clockwork-tfm | Not started (scaffolded) |

A GPT-style decoder-only transformer, packaged so other repos can install it. Grow it from the one-layer model in clockwork-tfm, following Karpathy's "Let's build GPT" video and nanoGPT.

## Goal

One readable, hookable backbone that trains on any integer token sequences: maze traces, game moves, text.

## Plan

- **Config (dataclass):** `vocab_size`, `n_layers`, `d_model`, `n_heads`, `d_mlp`, `max_seq_len`, `mlp` (gelu | relu | none for attention-only), `norm` (layernorm | rmsnorm | none), `dropout` (default 0).
- **Training:** causal mask, next-token loss at every position, and an optional loss mask (e.g. train only on the solution part of a maze sequence).
- **Hooks:** `return_cache=True` returns the residual stream before and after each attention and MLP layer, attention patterns and MLP neuron activations. A `patch` argument overwrites named activations for intervention experiments.
- **Generation:** `generate()` with greedy and temperature sampling, plus a hook that masks illegal tokens (legal moves, valid maze steps).
- **Utilities:** save and load, push to the Hugging Face Hub, parameter count.
- **Default size:** about 3M parameters (4 layers, d_model 256, 8 heads). Small enough to train in minutes on a laptop GPU.

## Tests

- Shapes, next-token alignment, padding and optional loss masking, including a batch
  with no unmasked targets.
- Causality: changing a later token never changes earlier logits.
- Overfitting a tiny batch and checking intended gradient paths.
- Cached and plain forward passes produce equivalent logits; identity patches preserve
  outputs, and invalid patch names or replacement shapes fail clearly.
- Saving and loading weights with their architecture config preserves predictions.
- Evaluation helpers restore the prior model mode even on failure. Gradient-based
  analysis can use eval mode with gradients enabled.

The implementation contract and scope are in
[CLAUDE.md](CLAUDE.md#project-specific-scope). These checks describe requirements for the
implementation; the current scaffold tests do not establish them yet.

## Done when

The definition of done every Latent Space Project repo shares:

Apply this checklist with the project-specific requirements and exceptions in
[CLAUDE.md](CLAUDE.md#project-specific-scope).

- `just quick` runs end to end on CPU in under 5 minutes, as a smoke test.
- The full run fits on one consumer GPU (8–12 GB VRAM) and auto-detects cuda, mps or cpu.
- Each run saves weights, `metrics.json` and its plots to `runs/<run-name>/`.
- This README shows the headline result with plots, compares against baselines, and states negative results plainly.
- Tests cover data generators, tensor shapes and one tiny overfit check.

## Notes

- **Optional:** a converter to TransformerLens's `HookedTransformer`, to borrow its analysis tools.
- Default plan: the owner writes this by hand following Karpathy, and Claude Code scaffolds tests and reviews (open question in the handoff doc).
- Other repos install this as a git dependency. While developing both at once, point them at the local folder with `uv add --editable ../tiny-transformer`, then switch back to the git source before committing.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Everything else, Python 3.12 included, comes from uv.

```sh
just setup              # uv sync, then install the pre-commit hooks
just quick              # CPU smoke test, under 5 minutes
just train              # full run; flags pass through, e.g. --run-name wd1.0-seed0
just analyze <run>      # baselines, probes, interventions and plots
just test               # pytest
just check              # ruff and basedpyright, as in CI
```

Copy `.env.example` to `.env` (gitignored) for `HF_TOKEN` and `HF_USER`.

## Layout

```
tiny-transformer/
├── README.md                goal, plan, results with plots
├── CLAUDE.md                instructions for coding agents; AGENTS.md points here
├── .agents/                 instructions/ (Python formatting, commits) and plans/
├── pyproject.toml           metadata, dependencies, tool settings
├── uv.lock                  committed lockfile
├── .python-version          pinned by uv
├── justfile                 every routine command
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/tiny_transformer/
│   └── config.py, model.py, generate.py, hub.py, utils.py
├── train.py                 CLI; --quick runs the CPU smoke test
├── predict.py               load weights and run inference
├── analyze.py               baselines, probes, interventions, plots
├── notebooks/               marimo notebooks for exploration
├── tests/
└── runs/                    outputs, gitignored
```

## Results

_No results yet._ The headline result with plots, the comparison against baselines, and any negative results go here.

## Sources

- nanoGPT
- TransformerLens

## License

MIT
