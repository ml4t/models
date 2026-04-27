"""Base classes for direct asset-prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ml4t.models.configs import AssetPredictionConfig
from ml4t.models.types import AssetSignalResult, CrossSectionBatch, FitSummary


class BaseAssetPredictionModel[ConfigT: AssetPredictionConfig](ABC):
    """Abstract base for direct asset-level predictive models."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def available_checkpoints(self) -> tuple[int, ...]:
        return ()

    @abstractmethod
    def fit(
        self,
        batch: CrossSectionBatch,
        *,
        validation_batch: CrossSectionBatch | None = None,
    ) -> FitSummary:
        """Fit the predictive model on aligned cross-sectional supervision."""

    @abstractmethod
    def predict(
        self,
        batch: CrossSectionBatch,
        *,
        checkpoint: int | None = None,
    ) -> AssetSignalResult:
        """Predict per-asset signals for each date in the batch."""

    def _mark_fitted(self) -> None:
        self._is_fitted = True
