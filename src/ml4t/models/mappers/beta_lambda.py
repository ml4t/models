"""Standard beta-times-lambda asset mapper."""

from __future__ import annotations

import numpy as np

from ml4t.models.mappers.base import BaseAssetMapper
from ml4t.models.types import AssetForecastResult, FactorForecastResult, LatentFactorState


class BetaLambdaMapper(BaseAssetMapper):
    """Map factor forecasts to asset returns via beta times factor premium."""

    def predict(
        self,
        state: LatentFactorState,
        factor_forecast: FactorForecastResult,
    ) -> AssetForecastResult:
        if state.n_periods != factor_forecast.factor_premia.shape[0]:
            raise ValueError(
                "state and factor forecast disagree on T: "
                f"expected {state.n_periods}, got {factor_forecast.factor_premia.shape[0]}"
            )
        if state.n_factors != factor_forecast.factor_premia.shape[1]:
            raise ValueError(
                "state and factor forecast disagree on K: "
                f"expected {state.n_factors}, got {factor_forecast.factor_premia.shape[1]}"
            )

        expected_returns = np.einsum(
            "tnk,tk->tn",
            state.asset_betas,
            factor_forecast.factor_premia,
            optimize=True,
        )
        return AssetForecastResult(
            expected_returns=expected_returns,
            timestamps=state.timestamps,
            asset_ids=state.asset_ids,
            metadata={"mapper": self.config.model_name},
        )
