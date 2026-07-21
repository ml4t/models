from __future__ import annotations

import numpy as np

from ml4t.models import CrossSectionBatch, IPCAConfig, IPCAModel


def test_ipca_recovers_single_factor_structure_on_ragged_cross_sections() -> None:
    rng = np.random.default_rng(7)
    n_periods = 12
    n_assets = 9
    n_features = 3

    characteristics = rng.normal(size=(n_periods, n_assets, n_features))
    gamma = np.array([[0.4], [0.8], [-0.5], [0.3]], dtype=np.float64)
    factors = rng.normal(scale=0.6, size=(n_periods, 1))
    augmented = np.concatenate(
        [np.ones((n_periods, n_assets, 1), dtype=np.float64), characteristics],
        axis=2,
    )
    betas = np.einsum("tnl,lk->tnk", augmented, gamma, optimize=True)
    returns = np.einsum("tnk,tk->tn", betas, factors, optimize=True)
    returns += 0.01 * rng.normal(size=returns.shape)

    batch = CrossSectionBatch(
        characteristics=characteristics,
        returns=returns,
        timestamps=tuple(f"2024-{month:02d}" for month in range(1, n_periods + 1)),
    )

    model = IPCAModel(IPCAConfig(n_factors=1, max_iter=200, tol=1e-8))
    fit = model.fit(batch)
    state = model.extract(batch)

    reconstructed = np.einsum("tnk,tk->tn", state.asset_betas, state.factor_returns, optimize=True)
    mse = float(np.nanmean((reconstructed - returns) ** 2))

    assert fit.train_metrics["train_mse"] < 1e-2
    assert state.asset_betas.shape == (n_periods, n_assets, 1)
    assert state.factor_returns is not None
    assert state.factor_returns.shape == (n_periods, 1)
    assert mse < 5e-3


def test_ipca_extracts_betas_without_future_returns() -> None:
    rng = np.random.default_rng(11)
    train = CrossSectionBatch(
        characteristics=rng.normal(size=(6, 5, 2)),
        returns=rng.normal(size=(6, 5)),
        mask=np.array(
            [
                [True, True, True, False, True],
                [True, True, False, False, True],
                [True, True, True, True, True],
                [True, False, True, True, True],
                [True, True, True, True, False],
                [True, True, True, True, True],
            ],
            dtype=bool,
        ),
    )
    future = CrossSectionBatch(
        characteristics=rng.normal(size=(3, 5, 2)),
        mask=np.array(
            [
                [True, False, True, True, True],
                [True, True, True, False, True],
                [True, True, True, True, True],
            ],
            dtype=bool,
        ),
    )

    model = IPCAModel(IPCAConfig(n_factors=1))
    model.fit(train)
    state = model.extract(future)

    assert state.factor_returns is None
    assert state.asset_betas.shape == (3, 5, 1)
    assert np.isnan(state.asset_betas[0, 1, 0])
    assert np.isfinite(state.asset_betas[0, 0, 0])


def test_ipca_recovers_multi_factor_structure() -> None:
    rng = np.random.default_rng(19)
    n_periods = 60
    n_assets = 30
    n_features = 6
    n_factors = 3
    characteristics = rng.normal(size=(n_periods, n_assets, n_features))
    gamma = rng.normal(scale=0.3, size=(n_features + 1, n_factors))
    factors = rng.normal(scale=0.5, size=(n_periods, n_factors))
    augmented = np.concatenate(
        [np.ones((n_periods, n_assets, 1), dtype=np.float64), characteristics],
        axis=2,
    )
    betas = np.einsum("tnl,lk->tnk", augmented, gamma, optimize=True)
    returns = np.einsum("tnk,tk->tn", betas, factors, optimize=True)
    returns += 0.01 * rng.normal(size=returns.shape)
    batch = CrossSectionBatch(characteristics=characteristics, returns=returns)
    model = IPCAModel(IPCAConfig(n_factors=n_factors, max_iter=400, tol=1e-6))

    fit = model.fit(batch)
    state = model.extract(batch)
    reconstructed = np.einsum("tnk,tk->tn", state.asset_betas, state.factor_returns, optimize=True)
    mse = float(np.nanmean((reconstructed - returns) ** 2))

    assert fit.converged
    assert mse < 5e-3
