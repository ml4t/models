"""Base classes for stochastic discount factor models."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ml4t.models.configs import SDFConfig
from ml4t.models.types import CrossSectionBatch, FitSummary, SDFState


class BaseSDFModel(ABC):
    """Abstract base for stochastic discount factor models."""

    def __init__(self, config: SDFConfig) -> None:
        self.config = config
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def available_checkpoints(self) -> tuple[int, ...]:
        return ()

    @abstractmethod
    def fit(self, batch: CrossSectionBatch) -> FitSummary:
        """Fit the SDF model on a dated cross-sectional batch."""

    @abstractmethod
    def extract(
        self,
        batch: CrossSectionBatch,
        *,
        checkpoint: int | None = None,
    ) -> SDFState:
        """Extract the structural SDF state from a batch."""

    def _mark_fitted(self) -> None:
        self._is_fitted = True
