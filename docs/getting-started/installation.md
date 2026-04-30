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

#### Compute device

Neural model configs include a `device` string (default `cpu`):

| Value               | Behavior                                                               |
| ------------------- | ---------------------------------------------------------------------- |
| `cpu`               | CPU tensors.                                                           |
| `cuda`, `cuda:0`, … | GPU when CUDA is available; otherwise CPU.                             |
| `mps`               | Apple Metal (MPS) when available in your PyTorch build; otherwise CPU. |

Example: `LSTMPortfolioConfig(..., device="mps")` on Apple Silicon with a suitable `torch` install.

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

Build the docs locally:

```bash
pip install ml4t-models[docs]
```

### Everything

```bash
pip install ml4t-models[all]
```

## Python Version

`ml4t-models` currently targets:

- Python `>=3.12,<3.14`

## Development Setup

Using `uv`:

```bash
git clone https://github.com/ml4t/models.git
cd ml4t-models
uv sync --all-extras
```

Run the quality gates:

```bash
uv run ruff check src/ tests/
uv run ty check
uv run pytest tests/ -q
uv build
```

Build the docs:

```bash
uv run --extra docs mkdocs build
```

## Related Libraries

`ml4t-models` is designed to integrate at boundaries with the rest of the ML4T stack:

- `ml4t-data` for dataset loading and canonical schema metadata
- `ml4t-engineer` for feature generation and labels
- `ml4t-diagnostic` for IC, validation, and report generation
- `ml4t-backtest` for execution and backtest state transitions
