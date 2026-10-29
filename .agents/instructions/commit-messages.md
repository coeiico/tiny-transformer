# Commit message instructions

Use Conventional Commits

<type>(<optional-scope>): <description>

Allowed types:

- feat
- fix
- docs
- style
- refactor
- perf
- test
- build
- ci
- chore
- revert

Scope examples:

- model
- data
- train
- analysis
- probes
- plots
- cli
- config
- deps
- tests
- docs
- build

Examples of valid output:
feat(model): add return_cache to the forward pass
fix(data): keep held-out mazes out of the training set
refactor(analysis): share probe fitting between layers
perf(train): move the full batch to the device once
test(model): check causality of the attention mask
docs(readme): add the grokking curve
chore: update dependencies
ci(build): cache dependencies

Examples of INVALID output:
Add refresh token support
Updated dependencies
Implement user authentication

Rules:

- Use lowercase for type and description.
- Use imperative mood.
- Do not capitalize the first word of the description.
- Do not end the subject line with a period.
- Prefer a single-line commit message.
