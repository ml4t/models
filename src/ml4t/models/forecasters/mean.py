"""Historical-mean factor-premium forecaster."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml4t.models._internal.observability import (
    observed_fit,
    restore_fit_record,
    state_with_fit_record,
)
from ml4t.models._internal.persistence import (
    load_artifact,
    load_config,
    require_array,
    require_array_names,
    save_artifact,
)
from ml4t.models.configs import ExpandingMeanForecasterConfig
from ml4t.models.forecasters.base import BaseFactorForecaster, require_estimable_factor_returns
from ml4t.models.types import FactorForecastResult, FitSummary, LatentFactorState


class ExpandingMeanFactorForecaster(BaseFactorForecaster[ExpandingMeanForecasterConfig]):
    """Forecast factor premia with the training-sample mean."""

    def __init__(self, config: ExpandingMeanForecasterConfig | None = None) -> None:
        super().__init__(config or ExpandingMeanForecasterConfig())
        self._mean_factor_premium: np.ndarray | None = None

    @observed_fit
    def fit(self, state: LatentFactorState) -> FitSummary:
        factors = require_estimable_factor_returns(state).astype(
            np.dtype(self.config.dtype), copy=False
        )
        mean_factor_premium = np.nanmean(factors, axis=0)
        if not np.isfinite(mean_factor_premium).all():
            raise FloatingPointError("factor mean estimation produced non-finite output")

        self._mean_factor_premium = mean_factor_premium
        self._mark_fitted()
        return FitSummary(
            converged=True,
            train_metrics={
                "n_train_periods": float(state.n_periods),
                "mean_abs_factor_premium": float(np.mean(np.abs(self._mean_factor_premium))),
            },
            notes=("Forecast uses the historical mean of fitted factor returns.",),
        )

    def predict(self, state: LatentFactorState) -> FactorForecastResult:
        if not self.is_fitted or self._mean_factor_premium is None:
            raise RuntimeError("Forecaster must be fitted before predict()")

        factor_premia = np.broadcast_to(
            self._mean_factor_premium[None, :],
            (state.n_periods, self._mean_factor_premium.shape[0]),
        ).copy()
        return FactorForecastResult(
            factor_premia=factor_premia,
            timestamps=state.timestamps,
            metadata={"model_name": self.config.model_name},
        )

    def save(self, path: str | Path) -> Path:
        if not self.is_fitted or self._mean_factor_premium is None:
            raise RuntimeError("ExpandingMeanFactorForecaster must be fitted before save()")
        return save_artifact(
            path,
            model_type="ml4t.models.ExpandingMeanFactorForecaster",
            config=self.config,
            state=state_with_fit_record(self, {}),
            arrays={"mean_factor_premium": self._mean_factor_premium},
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
    ) -> ExpandingMeanFactorForecaster:
        artifact = load_artifact(
            path,
            expected_model_type="ml4t.models.ExpandingMeanFactorForecaster",
        )
        require_array_names(artifact, {"mean_factor_premium"})
        model = cls(load_config(artifact, ExpandingMeanForecasterConfig, device=device))
        model._mean_factor_premium = require_array(
            artifact,
            "mean_factor_premium",
            ndim=1,
        )
        restore_fit_record(model, artifact.state)
        model._mark_fitted()
        return model
