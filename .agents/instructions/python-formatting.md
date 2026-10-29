# Python Formatting Guide

- Prefer to use single quotes over double quotes, except for docstrings (see below).
  Triple-quoted strings use double quotes as well, and a string that contains a single
  quote uses double quotes rather than escaping it.
- Use 2 spaces for indentation.
- Keep lines within 100 characters where you can; nothing enforces it (see Enforcement).

## Docstrings

- Use `"""triple double quotes"""` for all docstrings, never `'''`. Docstrings are the one
  exception to preferring single quotes.
- Single-line docstrings stay on one line: `"""Return the user's name."""`
- Multi-line docstrings put the opening `"""` on its own line, start the text on the next
  line, and put the closing `"""` on its own line:

  ```python
  def fetch(url, retries=3):
    """
    Fetch a URL and return the response body.

    Retries on transient network errors up to `retries` times.
    """
  ```

- The first line of text is a one-sentence summary ending in a period, followed by a blank
  line before any further detail.
- Docstring text is indented to match the opening quotes.
- Config dataclass fields are documented with a one-line docstring under the field; tyro
  turns it into that CLI flag's help text.
- Enforced by the `D` rules in `pyproject.toml` (`D213` on, `D212` off).

## Type Annotations

- Annotate function signatures. Annotate tensors with jaxtyping shapes
  (`Float[Tensor, 'batch seq d_model']`) wherever a shape matters: the tests check them at
  runtime with beartype.

## Enforcement

- `ruff check` enforces the quote rules (`Q`) and the docstring rules (`D`); the settings
  and the reasons for each ignored rule are in `pyproject.toml` under `[tool.ruff.lint]`.
- `ruff check` also rejects tab indentation (`W191`, `E101`). It cannot check the 2-space
  width: Ruff's only rule for it (`E111`) is preview-only and assumes 4. The formatter
  would hold it, but this repo deliberately does not run `ruff format`, so the width is
  kept by hand.
- `just check` runs `ruff check` and `basedpyright`, as CI and the pre-commit hook do.
  `just fix` applies Ruff's safe fixes.
