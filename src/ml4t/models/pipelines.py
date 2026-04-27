"""Composable pipelines for finance-native model workflows."""

from __future__ import annotations

from dataclasses import dataclass

from ml4t.models.api import (
    AssetMapper,
    FactorForecaster,
    LatentFactorModel,
    PanelBatch,
    PortfolioModel,
    PortfolioPostprocessor,
)
from ml4t.models.types import (
    FitSummary,
    LatentFactorPrediction,
    PortfolioPrediction,
    PortfolioSequenceBatch,
)


@dataclass(slots=True)
class PipelineFitResult:
    """Fit summaries for each stage of a latent-factor pipeline."""

    structural_fit: FitSummary
    factor_forecast_fit: FitSummary


@dataclass(slots=True)
class PortfolioPipelineFitResult:
    """Fit summaries for a portfolio-allocation pipeline."""

    model_fit: FitSummary


class LatentFactorForecastPipeline:
    """Compose structural extraction, factor forecasting, and asset mapping."""

    def __init__(
        self,
        model: LatentFactorModel,
        forecaster: FactorForecaster,
        mapper: AssetMapper,
    ) -> None:
        self.model = model
        self.forecaster = forecaster
        self.mapper = mapper

    def fit(self, batch: PanelBatch) -> PipelineFitResult:
        structural_fit = self.model.fit(batch)
        train_state = self.model.extract(batch)
        factor_forecast_fit = self.forecaster.fit(train_state)
        return PipelineFitResult(
            structural_fit=structural_fit,
            factor_forecast_fit=factor_forecast_fit,
        )

    def predict(
        self,
        batch: PanelBatch,
        *,
        checkpoint: int | None = None,
    ) -> LatentFactorPrediction:
        state = self.model.extract(batch, checkpoint=checkpoint)
        factor_forecast = self.forecaster.predict(state)
        asset_forecast = self.mapper.predict(state, factor_forecast)
        return LatentFactorPrediction(
            state=state,
            factor_forecast=factor_forecast,
            asset_forecast=asset_forecast,
        )


class PortfolioAllocationPipeline:
    """Compose a portfolio model with optional weight post-processing hooks."""

    def __init__(
        self,
        model: PortfolioModel,
        *,
        postprocessors: tuple[PortfolioPostprocessor, ...] = (),
    ) -> None:
        self.model = model
        self.postprocessors = postprocessors

    def fit(
        self,
        batch: PortfolioSequenceBatch,
        *,
        validation_batch: PortfolioSequenceBatch | None = None,
    ) -> PortfolioPipelineFitResult:
        model_fit = self.model.fit(batch, validation_batch=validation_batch)
        return PortfolioPipelineFitResult(model_fit=model_fit)

    def predict(
        self,
        batch: PortfolioSequenceBatch,
        *,
        checkpoint: int | None = None,
    ) -> PortfolioPrediction:
        raw_weights = self.model.predict(batch, checkpoint=checkpoint)
        processed_weights = raw_weights
        for postprocessor in self.postprocessors:
            processed_weights = postprocessor.transform(batch, processed_weights)
        return PortfolioPrediction(
            raw_weights=raw_weights,
            processed_weights=processed_weights,
        )
