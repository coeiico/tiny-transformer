set dotenv-load

hf_user := env('HF_USER', 'your-hf-username')
repo := file_name(justfile_directory())

default:
  @just --list

# Install dependencies and the pre-commit hooks
setup:
  uv sync
  uv run pre-commit install

# CPU smoke test, under 5 minutes
quick:
  uv run train.py --quick --run-name quick

# Full training run; pass flags through, e.g. just train --run-name wd1.0-seed0
train *args:
  uv run train.py {{args}}

# Inference on a saved run
predict run *args:
  uv run predict.py --run runs/{{run}} {{args}}

# Baselines, probes, interventions and plots for a saved run
analyze run:
  uv run analyze.py --run runs/{{run}}

test:
  uv run pytest

# Lint and type check, as in CI (no ruff format; see .agents/instructions/python-formatting.md)
check:
  uv run ruff check .
  uv run basedpyright

# Apply Ruff's safe fixes (import order, quotes, docstring layout)
fix:
  uv run ruff check --fix .

# Exploration notebooks (marimo, saved as plain .py files)
notebook name='explore':
  uv run marimo edit notebooks/{{name}}.py

# Upload a run's weights, metrics and plots to the Hugging Face Hub
push run:
  uv run hf upload {{hf_user}}/{{repo}} runs/{{run}}
