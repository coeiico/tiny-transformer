# General Formatting Guide

These rules apply to every file in this repository. `python-formatting.md` adds the
Python-specific rules on top of them.

## Indentation

- Whenever a language allows, use 2 spaces for indentation. That covers Python, the
  justfile, TOML, YAML, JSON, shell scripts, and lists and code blocks in Markdown.
- Never indent with tabs unless a format requires them.

## Quotes

- If there is no semantic difference between single quotes and double quotes, prefer
  single quotes over double quotes.
- Use double quotes where the two mean different things, or where the format allows
  nothing else:
  - **Python:** see `python-formatting.md`. Docstrings and other triple-quoted strings use
    double quotes, and so does a string that contains a single quote.
  - **justfile:** single-quoted strings are raw and double-quoted strings process escapes.
    Use single quotes unless the string needs an escape.
  - **TOML** (`pyproject.toml`): single-quoted strings are literal strings, with no escapes.
    Use single quotes unless the string needs an escape or contains a single quote.
  - **YAML:** leave plain values unquoted. When a value needs quoting, use single quotes
    unless it needs an escape.
  - **Shell:** single quotes stop `$VAR` and `$(...)` expansion, so use double quotes
    where you want the value expanded.
  - **JSON:** double quotes only.
- Generated files, such as `uv.lock`, keep whatever their tool writes. Don't reformat them.
