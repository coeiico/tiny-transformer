# Code Design

Machine learning code has two jobs: compute the right thing, and make it possible to know that it did. Most ML bugs don't crash. They produce a model that trains slightly worse, or a result that looks better than it is. So good ML code is simple enough to read against the method it implements, loud when something goes wrong, and careful about what its results actually show.

Most of what follows is a default: use judgement, and follow the project's own conventions where they already settle a question. The project's `CLAUDE.md` states its scope, exceptions and specific correctness checks. Those take precedence over defaults here. The rules under **Always** protect the integrity of the experiment across different methods.

## Always

- Keep final held-out evaluation independent of fitting and selection. Where there are validation and test splits, tune and select on validation, never on test. Document a different evaluation protocol when conventional splits do not fit the experiment.
- Keep related examples in the same split at the unit the generalisation claim requires. Do not use validation or test examples to fit preprocessing, models or probes; applying transformations fitted on training data to held-out inputs is expected.
- Stop a run on a NaN or infinite loss instead of training through it.
- Never silently ignore checkpoint mismatches. Strict loading is the default; intentional partial loading must check missing and unexpected keys against an explicit expected set and record what was loaded.
- Record the config, seed, code version and data version with every run's outputs.

## Evaluation and analysis

- For ordinary PyTorch validation and inference, use eval mode and disable gradient tracking. Model mode and gradient tracking are independent controls.
- Keep gradients enabled when the analysis needs derivatives, such as gradient-based attribution. Eval mode can still be appropriate; document any deliberate use of training-mode behaviour.
- A helper that temporarily changes model mode restores the previous mode, including deliberately frozen or evaluation-only submodules, even if it fails. Do not unconditionally switch the model to training afterwards.
- Fit probes on their training split, select their settings on validation, and report held-out performance. Freezing the backbone does not make probe fitting evaluation-only.

## Readable computation

- Write the computation so someone who knows the method can match the code to the maths. Favour clarity over cleverness in model and loss code.
- Keep the core training step (forward pass, loss, backward pass, parameter update) easy to find and follow.
- Name tensors and dimensions after what they mean, and note shapes where they aren't obvious.
- Use comments for the why: where a method comes from, why a constant has its value, what a numerical trick prevents.
- Keep file I/O, logging and device handling out of model and loss code where that's practical.
- Build what the current experiment needs. Add flexibility once there is a second real use for it.

## Silent failures

- Check the things that are wrong but still run: tensor shapes, broadcasting, mask direction, targets shifted by one, labels misaligned with inputs.
- Make sure gradients reach everything you mean to train: intended parameters are in the optimizer, and no accidental `detach` or in-place operation cuts the graph. A tiny overfit check is useful but does not prove every intended parameter receives gradients; inspect or test gradient paths where they matter.
- When code falls back to something (CPU instead of GPU, lower precision, a smaller dataset), say so.

## Data

- Before training on new data, print or plot a handful of examples with their labels. Check basic statistics such as value ranges, class balance and sequence lengths.
- Leakage often hides in near-duplicates, in related examples landing in different splits, or in random splits of time-ordered data. Split by source or by time where that applies.
- State the split unit, such as document, game, maze or trajectory. Keep derived windows, traces and augmentations with their source; record deliberate distribution shifts separately from in-distribution evaluation.
- Fit preprocessing statistics, such as normalisation and learned vocabularies, on training data only, then reuse them unchanged on held-out data. Fixed task vocabularies and externally pretrained tokenizers are allowed; record their definitions or versions.
- For single-target experiments or online environments, define independent evaluation conditions and the limits of the claim instead of inventing an inappropriate dataset split.
- Check external data when it's loaded, then rely on it.
- Make generated data reproducible from its settings and seed.

## Trustworthy results

- Compare against something: a simple baseline, a random or untrained model, an ablation. A number on its own says little.
- Define each metric in one place, so every report computes it the same way.
- When differences are small, check them across a few seeds before believing them.
- For probes and interventions, distinguish decodability from causal use. Compare interventions with identity, unrelated-direction and matched-magnitude controls as appropriate; measure the predicted effect and general performance degradation.
- Report what didn't work alongside what did, in the same place.

## Reproducible runs

- Seed every source of randomness, including data loader worker processes, not just the main process.
- Give each run its own output location. Refuse an existing run directory unless explicitly resuming that run; repeated smoke runs also need fresh directories. Save later analyses under their own names so they preserve the original run records.
- Record the resolved config, dependency lock identity, runtime versions, device and precision. Identify data by its revision or content hash and split definition; for generated data, include generator version, settings and seeds.
- A Git commit alone does not identify code with local changes. Record the commit and working-tree state, and preserve the relevant changes, including untracked source, or a source snapshot with the run. If version information is unavailable, say so.
- Save enough with each run (weights, config, metrics) to evaluate and analyse it later without retraining.
- Distinguish evaluation weights from a resumable checkpoint. The latter also holds optimizer and scheduler state, step count, random state and any other state needed by the method, such as a scaler, data position or sample pool. State whether resumption is exact or approximate.
- Make tests repeatable. Use exact equality for discrete invariants and controlled random sequences, and justified tolerances for floating-point results. Elsewhere reruns should agree within the measured noise.

## Configuration

- Put settings that affect results in configuration, not in constants buried in the code.
- Group settings by what they configure: model, data, optimisation.
- Start with a plain typed structure, such as a dataclass. Move to config files or layered composition only once plain structures become hard to manage.

## Abstraction

- Lean on the framework rather than building your own layer on top of it, except where implementing that mechanism is the project's learning objective.
- Avoid registries, factories and plugin systems until there are genuinely many interchangeable parts.
- A little duplication between scripts is fine. It is better than a shared helper that needs flags to serve every caller.

## Performance

- Start with small models and data, and scale up once the pipeline works end to end.
- Measure before optimising. When you do optimise, keep the simple version around as a reference and check they agree.

## Testing

- Useful defaults: data correctness, output shapes, a tiny model overfitting a tiny batch, and a quick end-to-end smoke run.
- Check losses and metrics against values worked out by hand on a tiny input.
- Check custom gradients against finite differences or a reference implementation.
- Choose checks for the method's actual claims: for example, causality, loss masking, cache and identity-patch equivalence, gradient reachability, or prediction equivalence after saving and loading. Shapes and tiny-batch overfitting alone do not establish these properties.
- A smoke run must exercise the implemented pipeline; a successful scaffold command is not evidence that training, evaluation or analysis works.
- Keep tests fast and small enough to run on a CPU.

## Changes

- Distinguish a change to the method being compared from a change to measurement. Training augmentations and model changes can be experimental variables under a fixed evaluation protocol. Changing metric meaning, evaluation preprocessing or the evaluated population requires reevaluating comparison runs under the same protocol, or clearly separating the results. A behaviour-preserving implementation change does not by itself invalidate comparison.
- Describe changes to the evaluation protocol wherever results are reported and in the commit message when a commit is requested.
- Change only what the task needs. Mention unrelated problems you notice rather than fixing them in the same change.
