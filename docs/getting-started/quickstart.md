# Quickstart: forecast returns from a small panel

This CPU workflow fits PCA to 12 months of synthetic returns, forecasts factor premia from the
training history, and maps them to six assets at two future dates. It uses the base installation:
`pip install ml4t-models`. No data download, credentials, or accelerator is needed.

Save the following as `first_forecast.py` and run `python first_forecast.py`:

```python
import numpy as np

from ml4t.models import (
    BetaLambdaMapper,
    ExpandingMeanFactorForecaster,
    LatentFactorForecastPipeline,
    PCAConfig,
    PCAModel,
    PersistentPanelBatch,
)

rng = np.random.default_rng(1)
asset_ids = tuple(f"asset_{i}" for i in range(6))
train = PersistentPanelBatch(
    returns=rng.normal(scale=0.02, size=(12, 6)),
    timestamps=tuple(f"2024-{month:02d}" for month in range(1, 13)),
    asset_ids=asset_ids,
)
future = PersistentPanelBatch(
    timestamps=("2025-01", "2025-02"),
    asset_ids=asset_ids,
)

pipeline = LatentFactorForecastPipeline(
    model=PCAModel(PCAConfig(n_factors=2)),
    forecaster=ExpandingMeanFactorForecaster(),
    mapper=BetaLambdaMapper(),
)
fit = pipeline.fit(train)
prediction = pipeline.predict(future)
forecast = prediction.asset_forecast.expected_returns

assert fit.structural_fit.converged
assert forecast.shape == (2, 6)
assert np.isfinite(forecast).all()
print(f"forecast shape: {forecast.shape}")
```

Expected output:

```text
forecast shape: (2, 6)
```

Each row is a future timestamp and each column is an asset in `asset_ids` order. These are model
forecasts from synthetic data, not evidence of trading performance. The forecaster uses the fitted
factor history; the future batch supplies dates and persistent asset identity without future returns.

The [latent-factor task guide](../user-guide/latent-factor-pipelines.md) explains the stages and
input requirements. See the exact
[`LatentFactorForecastPipeline` API](../api/index.md#pipelines) and the
[runnable repository example](https://github.com/ml4t/models/blob/main/examples/latent_factor_pipeline.py).

## Other supported tasks

| Task | Guide | Runnable example | Requirement |
|---|---|---|---|
| RP-PCA, IPCA, and CAE factor forecasts | [Latent-factor models](../user-guide/latent-factor-models.md) | [Bounded variant examples](https://github.com/ml4t/models/blob/main/examples/latent_factor_variants.py) | `deep` for CAE; production neural training can be long |
| SDF weights and optional return mapping | [SDF estimation](../user-guide/stochastic-discount-factor.md) | [CPU smoke example](https://github.com/ml4t/models/blob/main/examples/stochastic_discount_factor.py) | `ml4t-models[deep]`; production training can be long |
| Direct asset signals | [SAE prediction](../user-guide/direct-asset-prediction.md) | [CPU smoke example](https://github.com/ml4t/models/blob/main/examples/direct_asset_prediction.py) | `ml4t-models[deep]`; production training can be long |
| Portfolio weights | [Portfolio learning](../user-guide/portfolio-learning.md) | [Linear CPU example](https://github.com/ml4t/models/blob/main/examples/portfolio_learning.py) and [neural smoke example](https://github.com/ml4t/models/blob/main/examples/portfolio_neural.py) | Base install for the linear model; `deep` for LSTM and DeepPortfolio |
| Long-frame inputs and downstream frames | [Data contracts](../user-guide/data-contracts.md) and [Integration](../user-guide/integration.md) | [Adapter example](https://github.com/ml4t/models/blob/main/examples/integration_handoff.py) | `integration` extra for Parquet or Specs objects |

The [Book Guide](../book-guide/index.md) links to teaching notebooks and identifies where they
implement the methods manually. The [API Reference](../api/index.md) provides signatures and options.
