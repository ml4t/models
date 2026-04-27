"""Public config dataclasses."""

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
    SAEConfig,
    StochasticDiscountFactorConfig,
)
from ml4t.models.configs.pipeline import MapperConfig, PipelineConfig
from ml4t.models.configs.portfolio import DeepPortfolioConfig, LSTMPortfolioConfig, PortfolioConfig

__all__ = [
    "BaseModelConfig",
    "AR1ForecasterConfig",
    "CAEConfig",
    "DeepPortfolioConfig",
    "EWMABaseForecasterConfig",
    "ExpandingMeanForecasterConfig",
    "IPCAConfig",
    "LatentFactorConfig",
    "LSTMPortfolioConfig",
    "MapperConfig",
    "PCAConfig",
    "PipelineConfig",
    "PortfolioConfig",
    "SAEConfig",
    "StochasticDiscountFactorConfig",
]
