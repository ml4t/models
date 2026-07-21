from __future__ import annotations

import numpy as np

from ml4t.models import (
    LinearFeaturePortfolioModel,
    LinearPortfolioConfig,
    PortfolioAllocationPipeline,
    PortfolioSequenceBatch,
    WeightConstraintPostprocessor,
)

rng = np.random.default_rng(4)
n_windows, n_periods, n_assets, n_features = 3, 4, 5, 3
features = rng.normal(size=(n_windows, n_periods, n_assets, n_features))
returns = 0.03 * features[..., 0] - 0.01 * features[..., 1]
returns += 0.005 * rng.normal(size=returns.shape)

batch = PortfolioSequenceBatch(
    features=features,
    returns=returns,
    vol_scale=np.ones((n_windows, n_periods, n_assets), dtype=np.float64),
    mask=np.ones((n_windows, n_periods, n_assets), dtype=bool),
    asset_ids=tuple(f"asset_{idx}" for idx in range(n_assets)),
)
pipeline = PortfolioAllocationPipeline(
    LinearFeaturePortfolioModel(LinearPortfolioConfig(gross_exposure=1.0)),
    postprocessors=(WeightConstraintPostprocessor(gross_exposure=0.8),),
)
fit_result = pipeline.fit(batch)
prediction = pipeline.predict(batch)

assert fit_result.model_fit.converged
assert prediction.processed_weights.weights.shape == (n_windows, n_periods, n_assets)
assert np.all(np.abs(prediction.processed_weights.weights).sum(axis=-1) <= 0.8 + 1e-8)
