from __future__ import annotations

import numpy as np
import pytest

from ml4t.models import (
    BetaLambdaMapper,
    ExpandingMeanFactorForecaster,
    LatentFactorForecastPipeline,
    PCAConfig,
    PCAModel,
    PersistentPanelBatch,
)


def test_pca_pipeline_composes_structural_forecast_and_mapping() -> None:
    returns = np.array(
        [
            [0.02, 0.01, -0.01],
            [0.01, 0.03, -0.02],
            [0.00, 0.02, -0.01],
            [0.03, 0.01, -0.03],
        ],
        dtype=np.float64,
    )
    train = PersistentPanelBatch(
        returns=returns,
        timestamps=("2024-01", "2024-02", "2024-03", "2024-04"),
        asset_ids=("A", "B", "C"),
    )
    future = PersistentPanelBatch(
        timestamps=("2024-05", "2024-06"),
        asset_ids=("A", "B", "C"),
    )

    pipeline = LatentFactorForecastPipeline(
        model=PCAModel(PCAConfig(n_factors=2)),
        forecaster=ExpandingMeanFactorForecaster(),
        mapper=BetaLambdaMapper(),
    )
    fit_result = pipeline.fit(train)
    prediction = pipeline.predict(future)

    assert fit_result.structural_fit.converged
    assert fit_result.factor_forecast_fit.converged
    assert prediction.state.asset_betas.shape == (2, 3, 2)
    assert prediction.factor_forecast.factor_premia.shape == (2, 2)
    assert prediction.asset_forecast.expected_returns.shape == (2, 3)


def test_pca_extract_uses_training_factor_history_when_returns_are_available() -> None:
    returns = np.array(
        [
            [0.01, 0.02],
            [0.02, 0.01],
            [0.00, 0.03],
        ],
        dtype=np.float64,
    )
    batch = PersistentPanelBatch(returns=returns, asset_ids=("A", "B"))

    model = PCAModel(PCAConfig(n_factors=1))
    model.fit(batch)
    state = model.extract(batch)

    assert state.factor_returns is not None
    assert state.factor_returns.shape == (3, 1)


def test_pca_reorders_fitted_asset_state_by_identity() -> None:
    train = PersistentPanelBatch(
        returns=np.array([[1.0, 2.0], [3.0, 5.0], [4.0, 8.0]], dtype=np.float64),
        asset_ids=("A", "B"),
    )
    model = PCAModel(PCAConfig(n_factors=1))
    model.fit(train)

    original = model.extract(PersistentPanelBatch(timestamps=("t",), asset_ids=("A", "B")))
    reordered = model.extract(PersistentPanelBatch(timestamps=("t",), asset_ids=("B", "A")))

    assert np.allclose(reordered.asset_betas[:, 0], original.asset_betas[:, 1])
    assert np.allclose(reordered.asset_betas[:, 1], original.asset_betas[:, 0])


def test_pca_rejects_unknown_assets_and_checkpoints() -> None:
    model = PCAModel(PCAConfig(n_factors=1))
    model.fit(PersistentPanelBatch(returns=np.ones((3, 2)), asset_ids=("A", "B")))

    with pytest.raises(ValueError, match="asset_ids"):
        model.extract(PersistentPanelBatch(timestamps=("t",), asset_ids=("A", "C")))
    with pytest.raises(ValueError, match="checkpoints"):
        model.extract(PersistentPanelBatch(timestamps=("t",), asset_ids=("A", "B")), checkpoint=1)
