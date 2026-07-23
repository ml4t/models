# Quickstart

This quickstart shows the three main workflows in the library:

1. latent-factor forecasting
2. stochastic discount factor extraction
3. end-to-end portfolio learning

The same workflows are available as executable smoke examples in the repository's
`examples/` directory.

## 1. Latent-Factor Forecasting

The latent-factor path is intentionally three-stage:

1. fit a structural model
2. forecast factor premia
3. map those forecasts back to assets

```python
import numpy as np

from ml4t.models import (
    BetaLambdaMapper,
    CrossSectionBatch,
    ExpandingMeanFactorForecaster,
    IPCAConfig,
    IPCAModel,
    LatentFactorForecastPipeline,
)

batch = CrossSectionBatch(
    characteristics=np.random.randn(36, 250, 12),
    returns=np.random.randn(36, 250),
    timestamps=tuple(range(36)),
)

pipeline = LatentFactorForecastPipeline(
    model=IPCAModel(IPCAConfig(n_factors=3)),
    forecaster=ExpandingMeanFactorForecaster(),
    mapper=BetaLambdaMapper(),
)

fit_result = pipeline.fit(batch)
lf_prediction = pipeline.predict(batch)

print(fit_result.structural_fit.converged)
print(lf_prediction.state.asset_betas.shape)              # (36, 250, 3)
print(lf_prediction.factor_forecast.factor_premia.shape)  # (36, 3)
print(lf_prediction.asset_forecast.expected_returns.shape)
```

### Why This Matters

This separation matches the actual finance workflow:

- `IPCAModel` estimates conditional exposures and factor history
- `ExpandingMeanFactorForecaster` forecasts factor premia from that history
- `BetaLambdaMapper` computes asset-level expected returns

The same pipeline can be used with `PCAModel`, `RPPCAModel`, and `CAEModel`.

## 2. Weight-Native Stochastic Discount Factor

The stochastic discount factor family is different. It does not expose a `beta × lambda` forecast path as the native object.

```python
import numpy as np

from ml4t.models import CrossSectionBatch, StochasticDiscountFactorConfig, StochasticDiscountFactorModel

batch = CrossSectionBatch(
    characteristics=np.random.randn(48, 300, 16),
    returns=np.random.randn(48, 300),
    context_features=np.random.randn(48, 8),
    timestamps=tuple(range(48)),
)

config = StochasticDiscountFactorConfig(
    checkpoint_epochs=(256, 512, 768, 1024, 1280),
    default_checkpoint=1280,
)
model = StochasticDiscountFactorModel(config)
fit_summary = model.fit(batch)
state = model.extract(batch)

print(fit_summary.best_epoch)
print(state.asset_weights.shape)   # (48, 300)
print(state.sdf_values.shape)      # (48,)
```

Use this family when you want:

- no-arbitrage training
- weight-native outputs
- phase-aware checkpointed estimation

## 3. Direct Asset Prediction With SAE

`SAEModel` is treated as a direct predictor in this library.

```python
import numpy as np

from ml4t.models import CrossSectionBatch, SAEConfig, SAEModel

batch = CrossSectionBatch(
    characteristics=np.random.randn(24, 200, 20),
    returns=np.random.randn(24, 200),
    timestamps=tuple(range(24)),
)

model = SAEModel(SAEConfig(n_epochs=20, checkpoint_interval=5))
fit_summary = model.fit(batch, validation_batch=batch)
signals = model.predict(batch)

print(fit_summary.best_epoch)
print(signals.signal_values.shape)
```

## 4. End-To-End Portfolio Learning

Portfolio models learn weights directly.

```python
import numpy as np

from ml4t.models import LSTMPortfolioConfig, LSTMPortfolioModel, PortfolioSequenceBatch

batch = PortfolioSequenceBatch(
    features=np.random.randn(8, 63, 30, 10),
    returns=np.random.randn(8, 63, 30),
    timestamps=tuple(range(63)),
    asset_ids=tuple(f"asset_{i}" for i in range(30)),
)

model = LSTMPortfolioModel(
    LSTMPortfolioConfig(max_iters=20, checkpoint_every=5, default_checkpoint=20)
)
model.fit(batch, validation_batch=batch)
portfolio_prediction = model.predict(batch)

print(portfolio_prediction.weights.shape)
print(portfolio_prediction.checkpoint_step)
```

## 5. Export Frames For Backtesting And Diagnostics

```python
from ml4t.models import (
    backtest_inputs_from_asset_forecast,
    predictions_frame_from_asset_forecast,
    write_backtest_frames,
)

frame = predictions_frame_from_asset_forecast(forecast=lf_prediction.asset_forecast)
written = write_backtest_frames("artifacts/run_001", predictions=frame)

print(written["predictions"])
```

With the integration extra installed, you can also build a `DataFeed` handoff payload:

```python
inputs = backtest_inputs_from_asset_forecast(
    lf_prediction.asset_forecast,
    prices_path="prices.parquet",
    timestamp_col="timestamp",
    entity_col="asset",
    close_col="close",
)
```

## Next Steps

- [Data Contracts](../user-guide/data-contracts.md)
- [Latent-Factor Pipelines](../user-guide/latent-factor-pipelines.md)
- [Portfolio Learning](../user-guide/portfolio-learning.md)
- [Integration](../user-guide/integration.md)
