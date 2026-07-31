"""Base classes for factor forecasters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ml4t.models._internal.observability import FitObservable
from ml4t.models.configs.base import BaseModelConfig
from ml4t.models.types import FactorForecastResult, FitSummary, LatentFactorState


class BaseFactorForecaster[ConfigT: BaseModelConfig](FitObservable, ABC):
    """Abstract base for factor-premium forecasters."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self._is_fitted = False
        self._last_fit_record = None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def fit(self, state: LatentFactorState) -> FitSummary:
        """Fit on extracted training-state factor returns."""

    @abstractmethod
    def predict(self, state: LatentFactorState) -> FactorForecastResult:
        """Forecast factor premia for the target batch."""

    def _mark_fitted(self) -> None:
        self._is_fitted = True


def require_estimable_factor_returns(state: LatentFactorState) -> np.ndarray:
    if state.factor_returns is None:
        raise ValueError("factor forecaster requires training factor_returns")
    factors = np.asarray(state.factor_returns, dtype=np.float64)
    finite_counts = np.isfinite(factors).sum(axis=0)
    if np.any(finite_counts == 0):
        missing_factors = np.flatnonzero(finite_counts == 0).tolist()
        raise ValueError(
            "factor_returns require at least one finite value per factor; "
            f"factor positions {missing_factors} have none"
        )
    return factors
