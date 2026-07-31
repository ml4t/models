"""Base classes for factor-to-asset mappers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ml4t.models.configs.pipeline import MapperConfig
from ml4t.models.types import AssetForecastResult, FactorForecastResult, LatentFactorState


class BaseAssetMapper(ABC):
    """Abstract base for mapping factor forecasts to assets."""

    def __init__(self, config: MapperConfig | None = None) -> None:
        self.config = config or MapperConfig()

    @abstractmethod
    def predict(
        self,
        state: LatentFactorState,
        factor_forecast: FactorForecastResult,
    ) -> AssetForecastResult:
        """Map latent-factor forecasts to asset-level expected returns."""
