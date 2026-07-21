from __future__ import annotations

import numpy as np

from ml4t.models import CrossSectionBatch, SAEConfig, SAEModel

rng = np.random.default_rng(3)
n_periods, n_assets, n_features = 6, 6, 4
characteristics = rng.normal(size=(n_periods, n_assets, n_features))
returns = characteristics[..., 0] - 0.5 * characteristics[..., 1]
returns += 0.05 * rng.normal(size=returns.shape)

batch = CrossSectionBatch(
    characteristics=characteristics,
    returns=returns,
    timestamps=tuple(f"2024-{idx:02d}" for idx in range(1, n_periods + 1)),
    asset_ids=tuple(f"asset_{idx}" for idx in range(n_assets)),
)
model = SAEModel(
    SAEConfig(
        bottleneck_dim=4,
        aux_hidden_dim=4,
        main_hidden_units=(8, 8, 8, 8),
        dropout_rates=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        n_epochs=2,
        checkpoint_interval=2,
        batch_size=8,
        device="cpu",
    )
)
fit_result = model.fit(batch, validation_batch=batch)
signals = model.predict(batch)

assert fit_result.converged
assert signals.signal_values.shape == (n_periods, n_assets)
