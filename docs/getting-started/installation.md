# Installation

## Base Installation

```bash
pip install ml4t-models
```

`ml4t-models` keeps the base dependency set small. The default install gives you:

- typed batch and result contracts
- closed-form and NumPy-based model families
- pipeline composition utilities
- frame adapters that do not require heavy optional dependencies

## Optional Extras

### Neural Models

Install `torch`-backed models:

```bash
pip install ml4t-models[deep]
```

This extra is required for:

- `CAEModel`
- `SAEModel`
- `StochasticDiscountFactorModel`
- `LSTMPortfolioModel`
- `DeepPortfolioModel`

### Cross-Library Integration

Install tabular and spec helpers:

```bash
pip install ml4t-models[integration]
```

This extra is useful when you want:

- `ResultsFrame.to_polars()`
- parquet writing via `write_backtest_frames`
- `ml4t-specs`-aware schema resolution

### Documentation

Documentation tools are contributor dependencies. From a source checkout, install them with:

```bash
uv sync --extra docs
uv run mkdocs build --strict
```

### Everything

```bash
pip install ml4t-models[all]
```

The `all` extra installs only user-facing runtime capabilities from `deep` and `integration`. It
does not install test, lint, type-check, or documentation tools.

## Python Version

Stable releases support:

- Python 3.12
- Python 3.13
- Python 3.14

Python 3.15 prereleases run a separate compatibility gate. They are not part of the stable
support range until the Python and dependency ecosystems publish compatible stable releases.

## Development Setup

Using `uv`:

```bash
git clone https://github.com/ml4t/models.git
cd ml4t-models
uv sync --dev --extra docs
```

Run the quality gates:

```bash
uv run ruff check src/ tests/ examples/ scripts/
uv run ruff format --check src/ tests/ examples/ scripts/
uv run ty check
uv run pytest tests/ -q
uv build
```

Build the docs:

```bash
uv run mkdocs build --strict
```

## Related Libraries

`ml4t-models` is designed to integrate at boundaries with the rest of the ML4T stack:

- `ml4t-data` for dataset loading and canonical schema metadata
- `ml4t-engineer` for feature generation and labels
- `ml4t-diagnostic` for IC, validation, and report generation
- `ml4t-backtest` for execution and backtest state transitions
