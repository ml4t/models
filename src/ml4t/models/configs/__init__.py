"""Public config dataclasses."""

from ml4t.models.configs.asset_prediction import AssetPredictionConfig, SAEConfig
from ml4t.models.configs.base import BaseModelConfig
from ml4t.models.configs.forecast import (
    AR1ForecasterConfig,
    EWMABaseForecasterConfig,
    ExpandingMeanForecasterConfig,
)
from ml4t.models.configs.latent_factor import (
    CAEConfig,
    IPCAConfig,
    LatentFactorConfig,
    PCAConfig,
    RPPCAConfig,
    StochasticDiscountFactorConfig,
)
from ml4t.models.configs.pipeline import MapperConfig, PipelineConfig
from ml4t.models.configs.portfolio import (
    DeepPortfolioConfig,
    LinearPortfolioConfig,
    LSTMPortfolioConfig,
    PortfolioConfig,
)

__all__ = [
    "BaseModelConfig",
    "AR1ForecasterConfig",
    "AssetPredictionConfig",
    "CAEConfig",
    "DeepPortfolioConfig",
    "EWMABaseForecasterConfig",
    "ExpandingMeanForecasterConfig",
    "IPCAConfig",
    "LatentFactorConfig",
    "LinearPortfolioConfig",
    "LSTMPortfolioConfig",
    "MapperConfig",
    "PCAConfig",
    "PipelineConfig",
    "PortfolioConfig",
    "RPPCAConfig",
    "SAEConfig",
    "StochasticDiscountFactorConfig",
]
