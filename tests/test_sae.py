from __future__ import annotations

import numpy as np
import pytest

from ml4t.models import CrossSectionBatch, SAEConfig, SAEModel

pytest.importorskip("torch")


def test_sae_tracks_checkpoints_and_extracts_bottleneck_state() -> None:
    rng = np.random.default_rng(17)
    n_periods = 8
    n_assets = 9
    n_features = 3

    characteristics = rng.normal(size=(n_periods, n_assets, n_features))
    latent_beta = (
        0.3
        + 0.8 * characteristics[..., 0]
        - 0.5 * characteristics[..., 1]
        + 0.2 * characteristics[..., 2]
    )
    factor = np.linspace(-0.25, 0.35, num=n_periods, dtype=np.float64)[:, None]
    returns = latent_beta * factor
    returns += 0.01 * rng.normal(size=returns.shape)

    train = CrossSectionBatch(
        characteristics=characteristics,
        returns=returns,
        timestamps=tuple(f"2024-{month:02d}" for month in range(1, n_periods + 1)),
    )
    future = CrossSectionBatch(
        characteristics=rng.normal(size=(3, n_assets, n_features)),
        timestamps=("2024-09", "2024-10", "2024-11"),
    )

    model = SAEModel(
        SAEConfig(
            n_factors=1,
            hidden_units=(1, 8, 8, 8, 8, 8),
            dropout_rates=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            n_epochs=10,
            checkpoint_interval=5,
            n_ensemble=1,
            lr=1e-3,
        )
    )
    fit = model.fit(train)
    train_state = model.extract(train, checkpoint=5)
    future_state = model.extract(future, checkpoint=10)

    assert fit.converged
    assert model.available_checkpoints == (5, 10)
    assert fit.best_epoch == 10
    assert train_state.checkpoint_epoch == 5
    assert train_state.factor_returns is not None
    assert train_state.factor_returns.shape == (n_periods, 1)
    assert train_state.asset_betas.shape == (n_periods, n_assets, 1)
    assert future_state.checkpoint_epoch == 10
    assert future_state.factor_returns is None
    assert future_state.asset_betas.shape == (3, n_assets, 1)


def test_sae_requires_returns_for_training() -> None:
    batch = CrossSectionBatch(characteristics=np.zeros((2, 3, 2), dtype=np.float64))
    model = SAEModel(SAEConfig())

    with pytest.raises(ValueError):
        model.fit(batch)
