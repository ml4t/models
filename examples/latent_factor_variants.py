from __future__ import annotations

import numpy as np

from ml4t.models import (
    BetaLambdaMapper,
    CAEConfig,
    CAEModel,
    CrossSectionBatch,
    ExpandingMeanFactorForecaster,
    IPCAConfig,
    IPCAModel,
    LatentFactorForecastPipeline,
    PersistentPanelBatch,
    RPPCAConfig,
    RPPCAModel,
)

rng = np.random.default_rng(13)
asset_ids = tuple(f"A{i}" for i in range(7))
returns = rng.normal(scale=0.02, size=(8, 7))
panel = PersistentPanelBatch(
    returns=returns,
    timestamps=tuple(f"2024-{i:02d}" for i in range(1, 9)),
    asset_ids=asset_ids,
)
future_panel = PersistentPanelBatch(timestamps=("2024-09",), asset_ids=asset_ids)

rp_pipeline = LatentFactorForecastPipeline(
    model=RPPCAModel(RPPCAConfig(n_factors=1, gamma=1.0)),
    forecaster=ExpandingMeanFactorForecaster(),
    mapper=BetaLambdaMapper(),
)
rp_pipeline.fit(panel)
rp_forecast = rp_pipeline.predict(future_panel).asset_forecast.expected_returns
assert rp_forecast.shape == (1, 7) and np.isfinite(rp_forecast).all()

characteristics = rng.normal(size=(8, 7, 3))
returns = 0.03 * characteristics[..., 0] - 0.02 * characteristics[..., 1]
returns += 0.01 * rng.normal(size=returns.shape)
cross_sections = CrossSectionBatch(
    characteristics=characteristics,
    returns=returns,
    timestamps=panel.timestamps,
    asset_ids=asset_ids,
)
future_cross_section = CrossSectionBatch(
    characteristics=rng.normal(size=(1, 7, 3)),
    timestamps=("2024-09",),
    asset_ids=asset_ids,
)

for label, model in (
    ("IPCA", IPCAModel(IPCAConfig(n_factors=1, max_iter=30))),
    (
        "CAE",
        CAEModel(
            CAEConfig(
                n_factors=1,
                hidden_units=(4,),
                n_epochs=4,
                checkpoint_interval=2,
                batch_size=8,
            )
        ),
    ),
):
    pipeline = LatentFactorForecastPipeline(
        model=model,
        forecaster=ExpandingMeanFactorForecaster(),
        mapper=BetaLambdaMapper(),
    )
    pipeline.fit(cross_sections)
    forecast = pipeline.predict(future_cross_section).asset_forecast.expected_returns
    assert forecast.shape == (1, 7) and np.isfinite(forecast).all()
    print(f"{label} forecast shape: {forecast.shape}")

print(f"RP-PCA forecast shape: {rp_forecast.shape}")
