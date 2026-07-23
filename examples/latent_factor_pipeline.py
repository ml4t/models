from __future__ import annotations

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
returns = rng.normal(scale=0.02, size=(12, 6))
asset_ids = tuple(f"asset_{idx}" for idx in range(returns.shape[1]))

train = PersistentPanelBatch(
    returns=returns,
    timestamps=tuple(f"2024-{idx:02d}" for idx in range(1, returns.shape[0] + 1)),
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
fit_result = pipeline.fit(train)
prediction = pipeline.predict(future)

assert fit_result.structural_fit.converged
assert prediction.asset_forecast.expected_returns.shape == (2, 6)
