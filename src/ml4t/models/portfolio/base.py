"""Base classes for portfolio-learning models."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ml4t.models.configs import PortfolioConfig
from ml4t.models.types import FitSummary, PortfolioSequenceBatch, PortfolioWeightsResult


class BasePortfolioModel(ABC):
    """Abstract base for end-to-end portfolio learners."""

    def __init__(self, config: PortfolioConfig) -> None:
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
        batch: PortfolioSequenceBatch,
        *,
        validation_batch: PortfolioSequenceBatch | None = None,
    ) -> FitSummary:
        """Fit the portfolio model."""

    @abstractmethod
    def predict(
        self,
        batch: PortfolioSequenceBatch,
        *,
        checkpoint: int | None = None,
    ) -> PortfolioWeightsResult:
        """Emit implementable portfolio weights."""

    def _mark_fitted(self) -> None:
        self._is_fitted = True
