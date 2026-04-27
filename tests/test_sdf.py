from __future__ import annotations

import numpy as np
import pytest

from ml4t.models import CrossSectionBatch, LinearSDFReturnMapper, SDFConfig, SDFModel

pytest.importorskip("torch")


def test_sdf_extracts_checkpointed_weight_state_and_optional_return_mapping() -> None:
    rng = np.random.default_rng(23)
    n_periods = 8
    n_assets = 6
    n_features = 3
    n_context = 2

    characteristics = rng.normal(size=(n_periods, n_assets, n_features))
    context_features = rng.normal(size=(n_periods, n_context))
    signal = 0.4 * characteristics[..., 0] - 0.2 * characteristics[..., 1]
    signal += 0.1 * context_features[:, 0][:, None]
    returns = 0.05 * signal + 0.01 * rng.normal(size=signal.shape)

    train = CrossSectionBatch(
        characteristics=characteristics,
        returns=returns,
        context_features=context_features,
        timestamps=tuple(f"2024-{month:02d}" for month in range(1, n_periods + 1)),
        asset_ids=tuple(f"A{i}" for i in range(n_assets)),
    )
    future = CrossSectionBatch(
        characteristics=rng.normal(size=(3, n_assets, n_features)),
        context_features=rng.normal(size=(3, n_context)),
        timestamps=("2024-09", "2024-10", "2024-11"),
        asset_ids=tuple(f"A{i}" for i in range(n_assets)),
    )

    model = SDFModel(
        SDFConfig(
            state_dim_sdf=2,
            state_dim_moment=4,
            hidden_dim=8,
            n_instruments=3,
            n_epochs_unc=4,
            n_epochs_moment=2,
            n_epochs_cond=4,
            checkpoint_interval=2,
            lr=1e-3,
            dropout=0.0,
        )
    )
    fit = model.fit(train)
    train_state = model.extract(train, checkpoint=2)
    future_state = model.extract(future, checkpoint=8)

    mapper = LinearSDFReturnMapper()
    mapper_fit = mapper.fit(train_state, train)
    forecast = mapper.predict(future_state)

    assert fit.converged
    assert model.available_checkpoints == (2, 4, 6, 8)
    assert train_state.checkpoint_epoch == 2
    assert train_state.sdf_values is not None
    assert train_state.asset_weights.shape == (n_periods, n_assets)
    assert future_state.checkpoint_epoch == 8
    assert future_state.sdf_values is None
    assert future_state.asset_weights.shape == (3, n_assets)
    assert mapper_fit.converged
    assert forecast.expected_returns.shape == (3, n_assets)


def test_sdf_model_rejects_non_weight_native_output_mode() -> None:
    batch = CrossSectionBatch(
        characteristics=np.zeros((2, 3, 2), dtype=np.float64),
        returns=np.zeros((2, 3), dtype=np.float64),
    )
    model = SDFModel(SDFConfig(output_mode="expected_returns"))

    with pytest.raises(ValueError):
        model.fit(batch)
