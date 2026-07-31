"""Exponentially weighted factor-premium forecaster."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml4t.models._internal.persistence import (
    load_artifact,
    load_config,
    require_array,
    require_array_names,
    save_artifact,
)
from ml4t.models.configs import EWMABaseForecasterConfig
from ml4t.models.forecasters.base import BaseFactorForecaster
from ml4t.models.types import FactorForecastResult, FitSummary, LatentFactorState


class EWMABaseFactorForecaster(BaseFactorForecaster[EWMABaseForecasterConfig]):
    """Forecast factor premia with exponentially weighted moving averages."""

    def __init__(self, config: EWMABaseForecasterConfig | None = None) -> None:
        super().__init__(config or EWMABaseForecasterConfig())
        self._ewma_level: np.ndarray | None = None

    def fit(self, state: LatentFactorState) -> FitSummary:
        if state.factor_returns is None:
            raise ValueError("EWMABaseFactorForecaster requires training factor_returns")

        factors = np.asarray(state.factor_returns, dtype=np.float64)
        half_life = max(float(self.config.half_life), 1.0)
        alpha = 1.0 - np.exp(np.log(0.5) / half_life)

        level = np.nanmean(factors[:1], axis=0)
        for row in factors[1:]:
            values = np.where(np.isfinite(row), row, level)
            level = alpha * values + (1.0 - alpha) * level

        self._ewma_level = level
        self._mark_fitted()
        return FitSummary(
            converged=True,
            train_metrics={"half_life": half_life},
            notes=("EWMA level estimated from factor history.",),
        )

    def predict(self, state: LatentFactorState) -> FactorForecastResult:
        if not self.is_fitted or self._ewma_level is None:
            raise RuntimeError("Forecaster must be fitted before predict()")

        factor_premia = np.broadcast_to(
            self._ewma_level[None, :],
            (state.n_periods, self._ewma_level.shape[0]),
        ).copy()
        return FactorForecastResult(
            factor_premia=factor_premia,
            timestamps=state.timestamps,
            metadata={"model_name": self.config.model_name},
        )

    def save(self, path: str | Path) -> Path:
        if not self.is_fitted or self._ewma_level is None:
            raise RuntimeError("EWMABaseFactorForecaster must be fitted before save()")
        return save_artifact(
            path,
            model_type="ml4t.models.EWMABaseFactorForecaster",
            config=self.config,
            state={},
            arrays={"ewma_level": self._ewma_level},
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
    ) -> EWMABaseFactorForecaster:
        artifact = load_artifact(
            path,
            expected_model_type="ml4t.models.EWMABaseFactorForecaster",
        )
        require_array_names(artifact, {"ewma_level"})
        model = cls(load_config(artifact, EWMABaseForecasterConfig, device=device))
        model._ewma_level = require_array(artifact, "ewma_level", ndim=1)
        model._mark_fitted()
        return model
