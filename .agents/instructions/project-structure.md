# Project Structure

An ML repository holds things with very different lifecycles: code that defines the experiment, configuration that parameterises it, data it consumes, and outputs it produces. A good layout keeps these apart and makes it obvious where each part of the pipeline lives.

Treat what follows as defaults, not laws. Where the framework, the tooling, or the project's existing layout already settles a question, follow that. The project-specific scope in `CLAUDE.md` identifies the components and operations this experiment actually needs.

## Pipeline first, then the problem

- Organise importable code by the responsibilities the project needs: data, model, training, evaluation and analysis. These can start as modules within `src/<package>/`; they do not each require a directory, or an empty placeholder when unused.
- Within a stage, name modules after concepts from the problem domain (the tokenizer, the augmentations, the reward function) rather than generic roles.
- A domain concept that several stages use, such as a simulator, a game engine or an environment, gets its own module rather than living inside one of them.
- Prefer a specific name over a catch-all `utils` wherever one fits.
- Start flat. Split a module when it has grown to hold several distinct concepts, not just because it is long.

## Code, configuration, data and outputs

- Code and configuration go in version control.
- Large data stays out of version control. Regenerate data that can be regenerated, and keep downloaded data in an ignored location.
- Keep everything a run produces in that run's own directory, ignored by version control by default.
- Commit a result you want to show, such as a figure for the README, deliberately, and note which run produced it.

## Library code and entry points

- Keep reusable logic importable, so scripts, tests and notebooks all share the same code.
- Keep entry points thin: parse configuration and connect importable logic. Give distinct operations (training, evaluation, inference, analysis) independent commands or subcommands where needed, so you never retrain just to analyse.
- Notebooks are for exploration. Move code into the library once something depends on it, and make results that matter reproducible without the notebook.

## Dependencies

- A model shouldn't need to know where its data comes from or how it will be trained. Avoid imports between model implementations and data-loading code; put shared vocabulary definitions, schemas or domain types in a small module both can use. The entry point connects the components.
- Evaluation and analysis load saved weights and configuration through importable code, without importing an executable training script or requiring another training run. Their own settings and outputs remain identifiable separately from training.

## Experiments and variants

- Hyperparameters, seeds and scale belong in configuration; run names label those settings but do not replace the saved resolved config.
- Use config options for small variations. Give distinct algorithms or substantially different computations separate functions or modules within the same package when that makes them easier to read and test; avoid spreading variant flags throughout a shared implementation.
- A separate module is a much smaller decision than a separate repository. Create a new repo only for a genuinely independent project, not for an ablation or a different seed.

## Sharing code between projects

- Code that several projects use works best as a package they each depend on, not as imports by path.
- Pin shared packages to an identifiable version or revision. Local editable dependencies are useful during development; record their code state with runs too, and use a reproducible source when sharing the project.
- Copying a component early on is fine. Note where it came from, and promote it to a shared package once it settles down.

## Automation

- Script the routine commands (setup, training, evaluation, tests, a quick smoke run) so none of them depend on memory.
- Pin the environment (the dependency lockfile and the Python version) so results can be reproduced.

## Changing structure

- Don't reorganise working code for taste. Restructure when it's no longer clear where new code belongs, or when routine changes keep touching unrelated files.
