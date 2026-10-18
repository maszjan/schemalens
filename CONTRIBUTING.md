# Contributing

Thanks for helping. Issues and pull requests are welcome.

## Setup

```bash
uv sync
cp .env.example .env      # an LLM profile; tests need none
uv run pytest -q          # no network, no model
uv run ruff check . && uv run ruff format --check .
```

## Ground rules

- **All AI logic lives in `packages/core`** (plain Python, no web framework or cloud SDK). Adapters
  (`services/api`, the CLI, the eval runner) stay thin.
- **Models, keys and thresholds come from the environment** through `schemalens_core.config.Settings`,
  never hard-coded.
- **A missing value is `null`.** Do not add prompts that push the model to guess; fix values in code
  (normalizers, domain types) only when the result is unambiguous, and report every change in
  `warnings`.
- **Never log document content, OCR text or extracted values.** Metadata only.
- **Test data is synthetic.** Do not commit real documents with personal data; `samples/` is ignored.
- A new preset is a JSON file in `presets/` with a synthetic `example` (it becomes an eval case).
- After an API contract change: update `docs/api.md` and run `uv run python scripts/export_openapi.py`.
- Describe user-visible changes in `CHANGELOG.md` under `## [Unreleased]`.

## Evals

Quality changes (prompts, normalizers, confidence) should come with an eval run on the local model:

```bash
uv run python -m evals.run
```

By submitting a contribution you agree that it is licensed under the Apache License 2.0.
