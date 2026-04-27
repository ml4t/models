"""LSTM portfolio model scaffold."""

from __future__ import annotations

from ml4t.models.configs import LSTMPortfolioConfig
from ml4t.models.portfolio.base import BasePortfolioModel
from ml4t.models.types import FitSummary, PortfolioSequenceBatch, PortfolioWeightsResult


class LSTMPortfolioModel(BasePortfolioModel):
    """Placeholder for the first end-to-end portfolio learner."""

    def __init__(self, config: LSTMPortfolioConfig) -> None:
        super().__init__(config)
        self.config: LSTMPortfolioConfig = config

    def fit(
        self,
        batch: PortfolioSequenceBatch,
        *,
        validation_batch: PortfolioSequenceBatch | None = None,
    ) -> FitSummary:
        raise NotImplementedError("LSTM portfolio learner not implemented yet")

    def predict(
        self,
        batch: PortfolioSequenceBatch,
        *,
        checkpoint: int | None = None,
    ) -> PortfolioWeightsResult:
        raise NotImplementedError("LSTM portfolio learner not implemented yet")
