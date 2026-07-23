from __future__ import annotations

import numpy as np
import pytest

from ml4t.models import (
    BetaLambdaMapper,
    CAEConfig,
    CAEModel,
    CrossSectionBatch,
    ExpandingMeanFactorForecaster,
    LatentFactorForecastPipeline,
)
from ml4t.models._internal import cae_nn

torch = pytest.importorskip("torch")


def test_cae_tracks_available_checkpoints_and_supports_checkpointed_extract() -> None:
    rng = np.random.default_rng(5)
    n_periods = 8
    n_assets = 7
    n_features = 2

    characteristics = rng.normal(size=(n_periods, n_assets, n_features))
    beta = 0.2 + 0.7 * characteristics[..., 0] - 0.4 * characteristics[..., 1]
    factor = np.linspace(-0.3, 0.4, num=n_periods, dtype=np.float64)[:, None]
    returns = beta * factor
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

    model = CAEModel(
        CAEConfig(
            n_factors=1,
            hidden_units=(),
            n_epochs=10,
            checkpoint_interval=5,
            n_ensemble=1,
            lr=1e-2,
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
    assert train_state.asset_betas.shape == (n_periods, n_assets, 1)
    assert future_state.checkpoint_epoch == 10
    assert future_state.factor_returns is None
    assert future_state.asset_betas.shape == (3, n_assets, 1)


def test_cae_classification_requires_continuous_factor_returns() -> None:
    batch = CrossSectionBatch(
        characteristics=np.zeros((2, 3, 2), dtype=np.float64),
        returns=np.zeros((2, 3), dtype=np.float64),
    )

    model = CAEModel(CAEConfig(task_type="classification"))
    with pytest.raises(ValueError):
        model.fit(batch)


def test_cae_beta_network_uses_batch_norm() -> None:
    network = cae_nn.BetaNetwork(n_characteristics=4, n_factors=2, hidden_units=(8, 4))

    assert sum(isinstance(module, torch.nn.BatchNorm1d) for module in network.network) == 2


def test_cae_validation_batch_stores_default_val_best_checkpoint() -> None:
    rng = np.random.default_rng(7)
    n_periods = 8
    n_assets = 6
    n_features = 3
    characteristics = rng.normal(size=(n_periods, n_assets, n_features))
    returns = 0.05 * characteristics[..., 0] + 0.01 * rng.normal(size=(n_periods, n_assets))
    train = CrossSectionBatch(
        characteristics=characteristics,
        returns=returns,
        timestamps=tuple(f"2024-{month:02d}" for month in range(1, n_periods + 1)),
    )
    val = CrossSectionBatch(
        characteristics=rng.normal(size=(4, n_assets, n_features)),
        returns=rng.normal(scale=0.01, size=(4, n_assets)),
        timestamps=("2024-09", "2024-10", "2024-11", "2024-12"),
    )

    model = CAEModel(
        CAEConfig(
            n_factors=1,
            hidden_units=(4,),
            n_epochs=6,
            checkpoint_interval=3,
            n_ensemble=1,
            batch_size=8,
            lr=1e-2,
        )
    )
    fit = model.fit(train, validation_batch=val, patience=2)
    state = model.extract(val)

    assert 0 in model.available_checkpoints
    assert fit.best_epoch == 0
    assert fit.val_metrics["best_val_loss"] >= 0.0
    assert state.checkpoint_epoch == 0
    assert state.asset_betas.shape == (4, n_assets, 1)


def test_cae_extract_per_member_returns_member_states() -> None:
    rng = np.random.default_rng(9)
    n_periods = 6
    n_assets = 5
    n_features = 2
    characteristics = rng.normal(size=(n_periods, n_assets, n_features))
    returns = 0.02 * characteristics[..., 0] + 0.01 * rng.normal(size=(n_periods, n_assets))
    batch = CrossSectionBatch(characteristics=characteristics, returns=returns)

    model = CAEModel(
        CAEConfig(
            n_factors=1,
            hidden_units=(),
            n_epochs=4,
            checkpoint_interval=4,
            n_ensemble=2,
            lr=1e-2,
        )
    )
    model.fit(batch)
    states = model.extract_per_member(batch, checkpoint=4)

    assert len(states) == 2
    assert all(state.asset_betas.shape == (n_periods, n_assets, 1) for state in states)
    assert [state.metadata["ensemble_member"] for state in states] == [0, 1]


def test_cae_pipeline_forecasts_returns_through_checkpointed_two_step_path() -> None:
    rng = np.random.default_rng(13)
    n_periods = 8
    n_assets = 7
    n_features = 3
    characteristics = rng.normal(size=(n_periods, n_assets, n_features))
    returns = 0.03 * characteristics[..., 0] - 0.02 * characteristics[..., 1]
    returns += 0.01 * rng.normal(size=returns.shape)
    train = CrossSectionBatch(
        characteristics=characteristics,
        returns=returns,
        timestamps=tuple(f"2024-{month:02d}" for month in range(1, n_periods + 1)),
        asset_ids=tuple(f"A{idx}" for idx in range(n_assets)),
    )
    future = CrossSectionBatch(
        characteristics=rng.normal(size=(3, n_assets, n_features)),
        timestamps=("2024-09", "2024-10", "2024-11"),
        asset_ids=tuple(f"A{idx}" for idx in range(n_assets)),
    )
    pipeline = LatentFactorForecastPipeline(
        model=CAEModel(
            CAEConfig(
                n_factors=1,
                hidden_units=(4,),
                n_epochs=6,
                checkpoint_interval=3,
                n_ensemble=1,
                batch_size=8,
                lr=1e-2,
            )
        ),
        forecaster=ExpandingMeanFactorForecaster(),
        mapper=BetaLambdaMapper(),
    )

    fit = pipeline.fit(train)
    prediction = pipeline.predict(future, checkpoint=3)

    assert fit.structural_fit.converged
    assert prediction.state.factor_returns is None
    assert prediction.state.checkpoint_epoch == 3
    assert prediction.factor_forecast.factor_premia.shape == (3, 1)
    assert prediction.asset_forecast.expected_returns.shape == (3, n_assets)
    assert np.isfinite(prediction.asset_forecast.expected_returns).all()
