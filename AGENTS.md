# ml4t-models

Finance-native model implementations for latent-factor estimation, stochastic discount factor
learning, direct asset prediction, and end-to-end portfolio learning.

## Structure

| Directory | Purpose |
|-----------|---------|
| `src/ml4t/models/latent_factors/` | `PCAModel`, `RPPCAModel`, `IPCAModel`, `CAEModel` |
| `src/ml4t/models/stochastic_discount_factor/` | `StochasticDiscountFactorModel` (weight-native SDF) |
| `src/ml4t/models/asset_prediction/` | `SAEModel` (supervised autoencoder, direct predictor) |
| `src/ml4t/models/portfolio/` | `LinearFeaturePortfolioModel`, `LSTMPortfolioModel`, `DeepPortfolioModel` |
| `src/ml4t/models/forecasters/` | Factor-premium forecasters (e.g. `ExpandingMeanFactorForecaster`) |
| `src/ml4t/models/mappers/` | `BetaLambdaMapper` and other structural-to-asset mappers |
| `src/ml4t/models/configs/` | Per-model dataclass configs |
| `src/ml4t/models/integration/` | `predictions_frame_from_asset_forecast`, `write_backtest_frames` -> `ml4t-backtest`/`ml4t-diagnostic` handoff |
| `src/ml4t/models/pipelines.py` | `LatentFactorForecastPipeline` composing model + forecaster + mapper |
| `src/ml4t/models/types.py` | Finance-native batch contracts: `PersistentPanelBatch`, `CrossSectionBatch`, `PortfolioSequenceBatch` |
| `docs/` | MkDocs documentation |
| `docs/book-guide/index.md` | Chapter-to-API cross-reference: which book chapter/notebook maps to which model class |

## Entry Points

```python
from ml4t.models import (
    IPCAModel, IPCAConfig,
    StochasticDiscountFactorModel, StochasticDiscountFactorConfig,
    LSTMPortfolioModel, LSTMPortfolioConfig,
    LatentFactorForecastPipeline, ExpandingMeanFactorForecaster, BetaLambdaMapper,
    CrossSectionBatch, PersistentPanelBatch, PortfolioSequenceBatch,
    predictions_frame_from_asset_forecast, write_backtest_frames,
)
```

## Design Principles

- Finance-native data contracts rather than generic dataloaders.
- Explicit separation of stages: structural extraction -> factor-premium forecasting -> asset
  mapping -> downstream prediction/weight frames.
- Checkpoint-aware neural training (e.g. `checkpoint_epochs` on `StochasticDiscountFactorModel`).
- Integration with `ml4t-backtest` and `ml4t-diagnostic` at defined boundaries only, never by
  duplicating their evaluation logic.

## Documentation

- Book chapter-to-API cross-reference: [docs/book-guide/index.md](docs/book-guide/index.md) — start
  there instead of grepping source when mapping a book chapter/notebook to a model class.
- [Getting Started](docs/getting-started/quickstart.md)
- [User Guide](docs/user-guide/index.md)
- [Architecture](docs/reference/architecture.md)
- [API Reference](docs/api/index.md)
- Full detail and quick-start code for all four model families: `README.md`.

## Commands

```bash
uv sync
uv run pytest tests/ -q
uv run ty check
pre-commit run --all-files
```

## Version

Check `pyproject.toml` for current version.
PyPI: https://pypi.org/project/ml4t-models/
