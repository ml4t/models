"""Public package surface for ml4t-models."""

from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0b0"

from ml4t.models.api import (
    AssetMapper,
    AssetPredictionModel,
    FactorForecaster,
    LatentFactorModel,
    PortfolioModel,
    PortfolioPostprocessor,
    StochasticDiscountFactorEstimator,
)
from ml4t.models.asset_prediction import SAEModel
from ml4t.models.configs import (
    AR1ForecasterConfig,
    AssetPredictionConfig,
    CAEConfig,
    DeepPortfolioConfig,
    EWMABaseForecasterConfig,
    ExpandingMeanForecasterConfig,
    IPCAConfig,
    LatentFactorConfig,
    LinearPortfolioConfig,
    LSTMPortfolioConfig,
    MapperConfig,
    PCAConfig,
    PipelineConfig,
    PortfolioConfig,
    RPPCAConfig,
    SAEConfig,
    StochasticDiscountFactorConfig,
)
from ml4t.models.forecasters import (
    AR1FactorForecaster,
    EWMABaseFactorForecaster,
    ExpandingMeanFactorForecaster,
)
from ml4t.models.integration import (
    BacktestDataFeedInputs,
    ContextFrame,
    PredictionsFrame,
    ResolvedDatasetSchema,
    ResultsFrame,
    SignalsFrame,
    WeightsFrame,
    backtest_datafeed_inputs,
    backtest_inputs_from_asset_forecast,
    backtest_inputs_from_asset_signal,
    backtest_inputs_from_weights,
    context_frame_from_weights,
    cross_section_batch_from_long_frame,
    persistent_panel_batch_from_long_frame,
    predictions_frame_from_asset_forecast,
    predictions_frame_from_asset_signal,
    resolve_dataset_schema,
    resolve_feed_spec_mapping,
    signals_frame_from_asset_weights,
    signals_frame_from_portfolio_weights,
    weights_frame_from_asset_weights,
    weights_frame_from_portfolio_weights,
    write_backtest_frames,
)
from ml4t.models.latent_factors import CAEModel, IPCAModel, PCAModel, RPPCAModel
from ml4t.models.mappers import BetaLambdaMapper
from ml4t.models.pipelines import (
    LatentFactorForecastPipeline,
    PipelineFitResult,
    PortfolioAllocationPipeline,
    PortfolioPipelineFitResult,
)
from ml4t.models.stochastic_discount_factor import (
    LinearStochasticDiscountFactorReturnMapper,
    StochasticDiscountFactorBetaNetworkHead,
    StochasticDiscountFactorModel,
)
from ml4t.models.types import (
    AssetForecastResult,
    AssetSignalResult,
    AssetWeightsResult,
    CrossSectionBatch,
    FactorForecastResult,
    FitSummary,
    LatentFactorPrediction,
    LatentFactorState,
    PersistentPanelBatch,
    PortfolioPrediction,
    PortfolioSequenceBatch,
    PortfolioWeightsResult,
    StochasticDiscountFactorState,
)

__all__ = [
    "AssetForecastResult",
    "AR1FactorForecaster",
    "AR1ForecasterConfig",
    "AssetMapper",
    "AssetPredictionConfig",
    "AssetPredictionModel",
    "AssetSignalResult",
    "BetaLambdaMapper",
    "AssetWeightsResult",
    "BacktestDataFeedInputs",
    "CAEConfig",
    "CAEModel",
    "ContextFrame",
    "CrossSectionBatch",
    "backtest_datafeed_inputs",
    "backtest_inputs_from_asset_forecast",
    "backtest_inputs_from_asset_signal",
    "backtest_inputs_from_weights",
    "context_frame_from_weights",
    "DeepPortfolioConfig",
    "DeepPortfolioModel",
    "EWMABaseFactorForecaster",
    "EWMABaseForecasterConfig",
    "ExpandingMeanFactorForecaster",
    "ExpandingMeanForecasterConfig",
    "FactorForecaster",
    "FactorForecastResult",
    "FitSummary",
    "IPCAConfig",
    "IPCAModel",
    "LatentFactorConfig",
    "LatentFactorForecastPipeline",
    "LatentFactorModel",
    "LatentFactorPrediction",
    "LatentFactorState",
    "LinearFeaturePortfolioModel",
    "LinearPortfolioConfig",
    "LSTMPortfolioConfig",
    "LSTMPortfolioModel",
    "LinearStochasticDiscountFactorReturnMapper",
    "MapperConfig",
    "PCAConfig",
    "PCAModel",
    "PersistentPanelBatch",
    "PipelineConfig",
    "PipelineFitResult",
    "PortfolioAllocationPipeline",
    "PortfolioConfig",
    "PortfolioPipelineFitResult",
    "PortfolioPostprocessor",
    "PortfolioPrediction",
    "PortfolioModel",
    "PortfolioSequenceBatch",
    "PortfolioWeightsResult",
    "PredictionsFrame",
    "SAEConfig",
    "SAEModel",
    "ResolvedDatasetSchema",
    "ResultsFrame",
    "RPPCAConfig",
    "RPPCAModel",
    "SignalsFrame",
    "WeightsFrame",
    "cross_section_batch_from_long_frame",
    "persistent_panel_batch_from_long_frame",
    "predictions_frame_from_asset_forecast",
    "predictions_frame_from_asset_signal",
    "resolve_feed_spec_mapping",
    "resolve_dataset_schema",
    "signals_frame_from_asset_weights",
    "signals_frame_from_portfolio_weights",
    "StochasticDiscountFactorConfig",
    "StochasticDiscountFactorBetaNetworkHead",
    "StochasticDiscountFactorEstimator",
    "StochasticDiscountFactorModel",
    "StochasticDiscountFactorState",
    "weights_frame_from_portfolio_weights",
    "weights_frame_from_asset_weights",
    "WeightConstraintPostprocessor",
    "write_backtest_frames",
]


def __getattr__(name: str):
    module_map = {
        "DeepPortfolioModel": ("ml4t.models.portfolio.deep_portfolio", "DeepPortfolioModel"),
        "LinearFeaturePortfolioModel": (
            "ml4t.models.portfolio.linear",
            "LinearFeaturePortfolioModel",
        ),
        "LSTMPortfolioModel": ("ml4t.models.portfolio.lstm", "LSTMPortfolioModel"),
        "WeightConstraintPostprocessor": (
            "ml4t.models.portfolio.postprocessors",
            "WeightConstraintPostprocessor",
        ),
    }
    if name not in module_map:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = module_map[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
