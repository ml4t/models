"""Optional mappers from SDF weights to asset-level return forecasts."""

from __future__ import annotations

import numpy as np

from ml4t.models.types import AssetForecastResult, CrossSectionBatch, FitSummary, SDFState


class LinearSDFReturnMapper:
    """Map SDF weights to expected returns via a fitted linear projection."""

    def __init__(self) -> None:
        self._intercept = 0.0
        self._slope = 0.0
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, state: SDFState, batch: CrossSectionBatch) -> FitSummary:
        if batch.returns is None:
            raise ValueError("LinearSDFReturnMapper requires returns in the training batch")
        valid = np.isfinite(state.asset_weights) & np.isfinite(batch.returns)
        if valid.sum() < 2:
            self._intercept = 0.0
            self._slope = 0.0
        else:
            design = np.column_stack(
                [
                    np.ones(int(valid.sum()), dtype=np.float64),
                    state.asset_weights[valid].astype(np.float64),
                ]
            )
            coeffs, *_ = np.linalg.lstsq(
                design, batch.returns[valid].astype(np.float64), rcond=None
            )
            self._intercept = float(coeffs[0])
            self._slope = float(coeffs[1])
        self._is_fitted = True
        return FitSummary(
            converged=True,
            train_metrics={
                "intercept": self._intercept,
                "slope": self._slope,
            },
            notes=("Linear projection from SDF weights to expected returns.",),
        )

    def predict(self, state: SDFState) -> AssetForecastResult:
        if not self._is_fitted:
            raise RuntimeError("LinearSDFReturnMapper must be fitted before predict()")
        expected_returns = self._intercept + self._slope * state.asset_weights
        expected_returns = np.where(np.isfinite(state.asset_weights), expected_returns, np.nan)
        return AssetForecastResult(
            expected_returns=expected_returns,
            timestamps=state.timestamps,
            asset_ids=state.asset_ids,
            metadata={"mapper": "linear_sdf_return"},
        )
