"""Public protocols for ml4t-models."""

from __future__ import annotations

from typing import Protocol

from ml4t.models.types import (
    AssetForecastResult,
    CrossSectionBatch,
    FactorForecastResult,
    FitSummary,
    LatentFactorState,
    PersistentPanelBatch,
    PortfolioSequenceBatch,
    PortfolioWeightsResult,
    StochasticDiscountFactorState,
)

PanelBatch = PersistentPanelBatch | CrossSectionBatch


class LatentFactorModel(Protocol):
    """Protocol for structural latent-factor estimators."""

    is_fitted: bool

    def fit(self, batch: PanelBatch) -> FitSummary: ...

    def extract(
        self,
        batch: PanelBatch,
        *,
        checkpoint: int | None = None,
    ) -> LatentFactorState: ...


class FactorForecaster(Protocol):
    """Protocol for factor-premium forecasters."""

    is_fitted: bool

    def fit(self, state: LatentFactorState) -> FitSummary: ...

    def predict(self, state: LatentFactorState) -> FactorForecastResult: ...


class AssetMapper(Protocol):
    """Protocol for mapping factor forecasts back to asset forecasts."""

    def predict(
        self,
        state: LatentFactorState,
        factor_forecast: FactorForecastResult,
    ) -> AssetForecastResult: ...


class PortfolioModel(Protocol):
    """Protocol for end-to-end portfolio learners."""

    is_fitted: bool

    def fit(
        self,
        batch: PortfolioSequenceBatch,
        *,
        validation_batch: PortfolioSequenceBatch | None = None,
    ) -> FitSummary: ...

    def predict(
        self,
        batch: PortfolioSequenceBatch,
        *,
        checkpoint: int | None = None,
    ) -> PortfolioWeightsResult: ...


class StochasticDiscountFactorEstimator(Protocol):
    """Protocol for stochastic discount factor models with weight-native outputs."""

    is_fitted: bool

    def fit(self, batch: CrossSectionBatch) -> FitSummary: ...

    def extract(
        self,
        batch: CrossSectionBatch,
        *,
        checkpoint: int | None = None,
    ) -> StochasticDiscountFactorState: ...
