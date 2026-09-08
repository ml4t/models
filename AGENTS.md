# ml4t-models

Finance-specific models for latent-factor estimation, factor-premium forecasting, stochastic
discount factors, direct asset prediction, and end-to-end portfolio learning. The package separates
model estimation from forecasting and mapping so outputs can move explicitly into ML4T Backtest and
Diagnostic workflows.

## Public entry points

```python
from ml4t.models import CrossSectionBatch, IPCAModel, IPCAConfig
from ml4t.models import LatentFactorForecastPipeline, ExpandingMeanFactorForecaster
```

Use exports from `ml4t.models`. Torch-backed models load lazily and require the `deep` extra;
Polars and `ml4t-specs` integrations require the `integration` extra.

## Source map

| Path | Responsibility |
|---|---|
| `src/ml4t/models/api.py` | Protocols shared across model families |
| `src/ml4t/models/types.py` | Finance-native batches, fitted state, forecasts, and weights |
| `src/ml4t/models/latent_factors/` | PCA, risk-premium PCA, IPCA, and conditional autoencoders |
| `src/ml4t/models/forecasters/` | Ex ante factor-premium forecasting |
| `src/ml4t/models/mappers/` | Mapping structural outputs to asset forecasts |
| `src/ml4t/models/stochastic_discount_factor/` | Weight-native SDF estimation |
| `src/ml4t/models/asset_prediction/` | Direct asset prediction models |
| `src/ml4t/models/portfolio/` | Linear, recurrent, and deep portfolio models |
| `src/ml4t/models/integration/` | Data, backtest, diagnostic, and artifact boundaries |
| `scripts/ci/` | Release, coverage, performance, and hardware qualification |

## Modeling constraints

- Preserve temporal and cross-sectional identities in every batch and output object.
- Keep structural estimation, forecasting, asset mapping, and allocation as explicit stages.
- Make stochastic training reproducible and checkpoint semantics observable.
- Keep optional heavy dependencies out of the core import path.
- Record model-family assumptions in the relevant reference page and test financial invariants, not
  only array shapes.

## Quality commands

```bash
uv run ruff check src/ tests/ examples/ scripts/
uv run ruff format --check src/ tests/ examples/ scripts/
uv run ty check
uv run pytest tests/ -q --cov-report=json:coverage.json
uv run python scripts/ci/check_coverage.py coverage.json
uv run mkdocs build --strict
uv build
```
