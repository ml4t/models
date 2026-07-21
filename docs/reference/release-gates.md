# Release Gates

This page defines the checks required before tagging a beta or stable release.

## Local Gate

Run the full gate from the repository root:

```bash
uv run ruff check src/ tests/ examples
uv run ruff format --check src/ tests/ examples
uv run ty check
uv run pytest tests/ -q
uv run mkdocs build --strict
uv build
```

For optional integration paths, also run:

```bash
uv run --extra integration pytest \
  tests/test_integration_backtest.py \
  tests/test_integration_data.py \
  tests/test_integration_surfaces.py \
  -q
```

## GitHub Gate

The main CI workflow must pass before a beta tag is created:

- lint
- type check
- docs
- Python 3.12 tests
- Python 3.13 tests
- Python 3.14 tests
- package smoke test
- MPS smoke test
- package build

The release workflow publishes only from `v*` tags after the beta branch has merged.
