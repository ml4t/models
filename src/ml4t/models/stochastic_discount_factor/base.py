"""Base classes for stochastic discount factor models."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ml4t.models.configs import StochasticDiscountFactorConfig
from ml4t.models.types import CrossSectionBatch, FitSummary, StochasticDiscountFactorState

type SDFCheckpoint = tuple[str, int]


class BaseStochasticDiscountFactorModel(ABC):
    """Abstract base for stochastic discount factor models."""

    def __init__(self, config: StochasticDiscountFactorConfig) -> None:
        self.config = config
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def available_checkpoints(self) -> tuple[SDFCheckpoint, ...]:
        return ()

    @abstractmethod
    def fit(
        self,
        batch: CrossSectionBatch,
        *,
        validation_batch: CrossSectionBatch | None = None,
        patience: int | None = None,
    ) -> FitSummary:
        """Fit the stochastic discount factor model on a dated cross-sectional batch."""

    @abstractmethod
    def extract(
        self,
        batch: CrossSectionBatch,
        *,
        checkpoint: SDFCheckpoint | int | None = None,
    ) -> StochasticDiscountFactorState:
        """Extract the structural stochastic discount factor state from a batch."""

    def _mark_fitted(self) -> None:
        self._is_fitted = True
