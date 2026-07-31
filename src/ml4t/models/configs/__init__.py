"""Public config dataclasses."""

from ml4t.models.configs.asset_prediction import SAEConfig
from ml4t.models.configs.forecast import (
    AR1ForecasterConfig,
    EWMABaseForecasterConfig,
    ExpandingMeanForecasterConfig,
)
from ml4t.models.configs.latent_factor import (
    CAEConfig,
    IPCAConfig,
    PCAConfig,
    RPPCAConfig,
    StochasticDiscountFactorConfig,
)
from ml4t.models.configs.portfolio import (
    DeepPortfolioConfig,
    LinearPortfolioConfig,
    LSTMPortfolioConfig,
)

__all__ = [
    "AR1ForecasterConfig",
    "CAEConfig",
    "DeepPortfolioConfig",
    "EWMABaseForecasterConfig",
    "ExpandingMeanForecasterConfig",
    "IPCAConfig",
    "LinearPortfolioConfig",
    "LSTMPortfolioConfig",
    "PCAConfig",
    "RPPCAConfig",
    "SAEConfig",
    "StochasticDiscountFactorConfig",
]
