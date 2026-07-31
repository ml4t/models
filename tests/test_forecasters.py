from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml4t.models import (
    AR1FactorForecaster,
    AR1ForecasterConfig,
    EWMABaseFactorForecaster,
    EWMABaseForecasterConfig,
    ExpandingMeanFactorForecaster,
    LatentFactorState,
)


def test_ar1_forecaster_produces_recursive_factor_path() -> None:
    state = LatentFactorState(
        asset_betas=np.ones((4, 3, 1), dtype=np.float64),
        factor_returns=np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64),
        timestamps=("2024-01", "2024-02", "2024-03", "2024-04"),
    )
    future = LatentFactorState(
        asset_betas=np.ones((3, 2, 1), dtype=np.float64),
        timestamps=("2024-05", "2024-06", "2024-07"),
    )

    forecaster = AR1FactorForecaster(AR1ForecasterConfig())
    forecaster.fit(state)
    forecast = forecaster.predict(future)

    assert forecast.factor_premia.shape == (3, 1)
    assert forecast.factor_premia[0, 0] < forecast.factor_premia[1, 0]
    assert forecast.factor_premia[1, 0] <= forecast.factor_premia[2, 0]


def test_ewma_forecaster_broadcasts_last_level() -> None:
    state = LatentFactorState(
        asset_betas=np.ones((5, 2, 2), dtype=np.float64),
        factor_returns=np.array(
            [
                [0.1, -0.2],
                [0.2, -0.1],
                [0.3, 0.0],
                [0.4, 0.1],
                [0.5, 0.2],
            ],
            dtype=np.float64,
        ),
    )
    future = LatentFactorState(asset_betas=np.ones((2, 2, 2), dtype=np.float64))

    forecaster = EWMABaseFactorForecaster(EWMABaseForecasterConfig(half_life=2.0))
    forecaster.fit(state)
    forecast = forecaster.predict(future)

    assert forecast.factor_premia.shape == (2, 2)
    assert np.allclose(forecast.factor_premia[0], forecast.factor_premia[1])


@pytest.mark.parametrize(
    "forecaster",
    [
        AR1FactorForecaster(),
        EWMABaseFactorForecaster(),
        ExpandingMeanFactorForecaster(),
    ],
)
def test_forecasters_reject_prediction_and_save_before_fit(forecaster: Any, tmp_path: Path) -> None:
    state = LatentFactorState(asset_betas=np.ones((2, 2, 1)))

    with pytest.raises(RuntimeError, match="fitted before predict"):
        forecaster.predict(state)
    with pytest.raises(RuntimeError, match="fitted before save"):
        forecaster.save(tmp_path / "forecast.ml4t")
