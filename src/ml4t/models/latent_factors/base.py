"""Base classes for latent-factor estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ml4t.models.api import PanelBatch
from ml4t.models.configs.latent_factor import LatentFactorConfig
from ml4t.models.types import FitSummary, LatentFactorState


class BaseLatentFactorModel[ConfigT: LatentFactorConfig](ABC):
    """Abstract base for structural latent-factor estimators."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def fit(self, batch: PanelBatch) -> FitSummary:
        """Estimate latent-factor structure from a training batch."""

    @abstractmethod
    def extract(
        self,
        batch: PanelBatch,
        *,
        checkpoint: int | None = None,
    ) -> LatentFactorState:
        """Extract factor state for a batch using fitted model parameters."""

    def _mark_fitted(self) -> None:
        self._is_fitted = True

    @property
    def available_checkpoints(self) -> tuple[int, ...]:
        return ()
