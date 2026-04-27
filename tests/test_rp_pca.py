from __future__ import annotations

import numpy as np

from ml4t.models import PersistentPanelBatch, RPPCAConfig, RPPCAModel


def test_rp_pca_extracts_static_betas_and_factor_history() -> None:
    rng = np.random.default_rng(13)
    n_periods = 12
    n_assets = 6
    loadings = rng.normal(size=(n_assets, 2))
    factor_returns = np.column_stack(
        [
            np.linspace(0.02, 0.10, num=n_periods),
            np.sin(np.linspace(0.0, 2.0, num=n_periods)) * 0.03,
        ]
    )
    returns = factor_returns @ loadings.T
    returns += 0.01 * rng.normal(size=returns.shape)

    batch = PersistentPanelBatch(
        returns=returns,
        timestamps=tuple(f"2024-{month:02d}" for month in range(1, n_periods + 1)),
        asset_ids=tuple(f"A{i}" for i in range(n_assets)),
    )
    model = RPPCAModel(
        RPPCAConfig(
            n_factors=2,
            gamma=5.0,
            base_moment="covariance",
        )
    )

    fit = model.fit(batch)
    state = model.extract(batch)

    assert fit.converged
    assert state.factor_returns is not None
    assert state.factor_returns.shape == (n_periods, 2)
    assert state.asset_betas.shape == (n_periods, n_assets, 2)
    assert state.metadata["gamma"] == 5.0


def test_rp_pca_supports_second_moment_parameterization() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(size=(10, 5))
    batch = PersistentPanelBatch(returns=returns)
    model = RPPCAModel(
        RPPCAConfig(
            n_factors=2,
            gamma=-1.0,
            base_moment="second_moment",
            scale_by_asset_volatility=True,
        )
    )

    fit = model.fit(batch)
    state = model.extract(batch)

    assert fit.converged
    assert state.factor_returns is not None
    assert np.isfinite(state.factor_returns).all()
