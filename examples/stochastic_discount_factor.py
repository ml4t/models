from __future__ import annotations

import numpy as np

from ml4t.models import (
    CrossSectionBatch,
    LinearStochasticDiscountFactorReturnMapper,
    StochasticDiscountFactorConfig,
    StochasticDiscountFactorModel,
)

rng = np.random.default_rng(2)
n_periods, n_assets, n_features = 6, 5, 3
characteristics = rng.normal(size=(n_periods, n_assets, n_features))
returns = 0.04 * characteristics[..., 0] - 0.02 * characteristics[..., 1]
returns += 0.01 * rng.normal(size=returns.shape)

batch = CrossSectionBatch(
    characteristics=characteristics,
    returns=returns,
    timestamps=tuple(f"2024-{idx:02d}" for idx in range(1, n_periods + 1)),
    asset_ids=tuple(f"asset_{idx}" for idx in range(n_assets)),
)
model = StochasticDiscountFactorModel(
    StochasticDiscountFactorConfig(
        state_dim_sdf=2,
        state_dim_moment=4,
        hidden_dim=8,
        n_instruments=3,
        n_epochs_unc=2,
        n_epochs_moment=1,
        n_epochs_cond=2,
        checkpoint_interval=2,
        dropout=0.0,
        device="cpu",
    )
)
fit_result = model.fit(batch)
state = model.extract(batch)
mapper = LinearStochasticDiscountFactorReturnMapper()
mapper.fit(state, batch)
forecast = mapper.predict(state)

assert fit_result.converged
assert state.asset_weights.shape == (n_periods, n_assets)
assert forecast.expected_returns.shape == (n_periods, n_assets)
