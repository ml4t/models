"""Per-factor autoregressive forecaster."""

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
from ml4t.models.configs import AR1ForecasterConfig
from ml4t.models.forecasters.base import BaseFactorForecaster, require_estimable_factor_returns
from ml4t.models.types import FactorForecastResult, FitSummary, LatentFactorState


class AR1FactorForecaster(BaseFactorForecaster[AR1ForecasterConfig]):
    """Forecast factor premia with independent AR(1) models."""

    def __init__(self, config: AR1ForecasterConfig | None = None) -> None:
        super().__init__(config or AR1ForecasterConfig())
        self._intercepts: np.ndarray | None = None
        self._slopes: np.ndarray | None = None
        self._last_values: np.ndarray | None = None
        self._fallback_mean: np.ndarray | None = None

    def fit(self, state: LatentFactorState) -> FitSummary:
        dtype = np.dtype(self.config.dtype)
        factors = require_estimable_factor_returns(state).astype(dtype, copy=False)
        n_periods, n_factors = factors.shape
        fallback_mean = np.nanmean(factors, axis=0)
        last_values = np.asarray(
            [factor[np.isfinite(factor)][-1] for factor in factors.T],
            dtype=dtype,
        )
        intercepts = np.zeros(n_factors, dtype=dtype)
        slopes = np.zeros(n_factors, dtype=dtype)

        if n_periods < 2:
            intercepts = fallback_mean.copy()
            self._fallback_mean = fallback_mean
            self._last_values = last_values
            self._intercepts = intercepts
            self._slopes = slopes
            self._mark_fitted()
            return FitSummary(
                converged=True,
                train_metrics={"n_train_periods": float(n_periods)},
                notes=("Insufficient history for AR(1); using mean fallback.",),
            )

        x = factors[:-1]
        y = factors[1:]
        for factor_idx in range(n_factors):
            x_k = x[:, factor_idx]
            y_k = y[:, factor_idx]
            valid = np.isfinite(x_k) & np.isfinite(y_k)
            if valid.sum() < 2:
                intercepts[factor_idx] = fallback_mean[factor_idx]
                continue
            design = np.column_stack([np.ones(valid.sum(), dtype=dtype), x_k[valid]])
            coeffs, *_ = np.linalg.lstsq(design, y_k[valid], rcond=None)
            intercepts[factor_idx], slopes[factor_idx] = coeffs

        fitted_values = (intercepts, slopes, last_values, fallback_mean)
        if not all(np.isfinite(value).all() for value in fitted_values):
            raise FloatingPointError("AR(1) estimation produced non-finite output")

        self._fallback_mean = fallback_mean
        self._last_values = last_values
        self._intercepts = intercepts
        self._slopes = slopes
        self._mark_fitted()
        return FitSummary(
            converged=True,
            train_metrics={"n_train_periods": float(n_periods)},
            notes=("Independent AR(1) fitted per factor.",),
        )

    def predict(self, state: LatentFactorState) -> FactorForecastResult:
        if not self.is_fitted:
            raise RuntimeError("Forecaster must be fitted before predict()")
        assert self._intercepts is not None
        assert self._slopes is not None
        assert self._last_values is not None
        assert self._fallback_mean is not None

        n_periods = state.n_periods
        forecasts = np.zeros(
            (n_periods, self._intercepts.shape[0]), dtype=np.dtype(self.config.dtype)
        )
        previous = self._last_values.copy()
        for step in range(n_periods):
            next_values = self._intercepts + self._slopes * previous
            next_values = np.where(np.isfinite(next_values), next_values, self._fallback_mean)
            forecasts[step] = next_values
            previous = next_values

        return FactorForecastResult(
            factor_premia=forecasts,
            timestamps=state.timestamps,
            metadata={"model_name": self.config.model_name},
        )

    def save(self, path: str | Path) -> Path:
        if (
            not self.is_fitted
            or self._intercepts is None
            or self._slopes is None
            or self._last_values is None
            or self._fallback_mean is None
        ):
            raise RuntimeError("AR1FactorForecaster must be fitted before save()")
        return save_artifact(
            path,
            model_type="ml4t.models.AR1FactorForecaster",
            config=self.config,
            state={},
            arrays={
                "intercepts": self._intercepts,
                "slopes": self._slopes,
                "last_values": self._last_values,
                "fallback_mean": self._fallback_mean,
            },
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> AR1FactorForecaster:
        artifact = load_artifact(path, expected_model_type="ml4t.models.AR1FactorForecaster")
        names = {"intercepts", "slopes", "last_values", "fallback_mean"}
        require_array_names(artifact, names)
        model = cls(load_config(artifact, AR1ForecasterConfig, device=device))
        model._intercepts = require_array(artifact, "intercepts", ndim=1)
        model._slopes = require_array(artifact, "slopes", ndim=1)
        model._last_values = require_array(artifact, "last_values", ndim=1)
        model._fallback_mean = require_array(artifact, "fallback_mean", ndim=1)
        if (
            len(
                {
                    array.shape
                    for array in (
                        model._intercepts,
                        model._slopes,
                        model._last_values,
                        model._fallback_mean,
                    )
                }
            )
            != 1
        ):
            raise ValueError("artifact AR(1) state dimensions disagree")
        model._mark_fitted()
        return model
