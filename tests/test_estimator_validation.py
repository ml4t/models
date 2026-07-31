from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from ml4t.models import (
    AR1FactorForecaster,
    EWMABaseFactorForecaster,
    ExpandingMeanFactorForecaster,
    LatentFactorState,
    PCAConfig,
    PCAModel,
    PersistentPanelBatch,
)


def _factor_state(factor_returns: np.ndarray) -> LatentFactorState:
    return LatentFactorState(
        asset_betas=np.ones((factor_returns.shape[0], 2, factor_returns.shape[1])),
        factor_returns=factor_returns,
    )


def test_pca_rejects_panel_without_finite_information() -> None:
    for returns in (
        np.full((4, 3), np.nan),
        np.ones((4, 3)),
    ):
        model = PCAModel(PCAConfig(n_factors=1))

        with pytest.raises(ValueError, match="PCA requires"):
            model.fit(PersistentPanelBatch(returns=returns))

        assert not model.is_fitted


def test_pca_rejects_unavailable_configured_factor_count() -> None:
    model = PCAModel(PCAConfig(n_factors=3))

    with pytest.raises(ValueError, match="n_factors=3 exceeds"):
        model.fit(PersistentPanelBatch(returns=np.arange(8, dtype=np.float64).reshape(4, 2)))


@pytest.mark.parametrize(
    "factory",
    [
        AR1FactorForecaster,
        EWMABaseFactorForecaster,
        ExpandingMeanFactorForecaster,
    ],
)
def test_forecasters_reject_factor_without_finite_history(
    factory: Callable[[], object],
) -> None:
    forecaster = factory()
    state = _factor_state(np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]]))

    with pytest.raises(ValueError, match="factor_returns require at least one finite value"):
        forecaster.fit(state)

    assert not forecaster.is_fitted


@pytest.mark.parametrize(
    "factory",
    [
        AR1FactorForecaster,
        EWMABaseFactorForecaster,
        ExpandingMeanFactorForecaster,
    ],
)
def test_forecasters_support_partially_missing_finite_histories(
    factory: Callable[[], object],
) -> None:
    forecaster = factory()
    train = _factor_state(
        np.array(
            [
                [np.nan, 1.0],
                [2.0, np.nan],
                [3.0, 2.0],
            ]
        )
    )
    future = LatentFactorState(asset_betas=np.ones((2, 2, 2)))

    forecaster.fit(train)
    forecast = forecaster.predict(future)

    assert np.isfinite(forecast.factor_premia).all()
