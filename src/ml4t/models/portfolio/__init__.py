"""Portfolio-learning model family."""

from __future__ import annotations

from importlib import import_module

from ml4t.models.portfolio.base import BasePortfolioModel

__all__ = [
    "BasePortfolioModel",
    "DeepPortfolioModel",
    "LinearFeaturePortfolioModel",
    "LSTMPortfolioModel",
    "PortfolioLossOutput",
    "WeightConstraintPostprocessor",
    "robust_sharpe_loss",
]


def __getattr__(name: str):
    module_map = {
        "DeepPortfolioModel": ("ml4t.models.portfolio.deep_portfolio", "DeepPortfolioModel"),
        "LinearFeaturePortfolioModel": ("ml4t.models.portfolio.linear", "LinearFeaturePortfolioModel"),
        "LSTMPortfolioModel": ("ml4t.models.portfolio.lstm", "LSTMPortfolioModel"),
        "PortfolioLossOutput": ("ml4t.models.portfolio.losses", "PortfolioLossOutput"),
        "robust_sharpe_loss": ("ml4t.models.portfolio.losses", "robust_sharpe_loss"),
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
