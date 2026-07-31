"""Tensor-independent validation for portfolio model batches."""

from __future__ import annotations

from ml4t.models.configs.portfolio import PortfolioConfig
from ml4t.models.types import PortfolioSequenceBatch


def validate_portfolio_training_batch(
    batch: PortfolioSequenceBatch,
    config: PortfolioConfig | None = None,
) -> None:
    if batch.returns is None:
        raise ValueError("portfolio training requires forward returns in the batch")
    if config is not None:
        _validate_context(batch, config)


def validate_portfolio_prediction_batch(
    batch: PortfolioSequenceBatch,
    config: PortfolioConfig,
) -> None:
    _validate_context(batch, config)


def validate_portfolio_identity(
    batch: PortfolioSequenceBatch,
    fitted_asset_ids: tuple[str, ...],
) -> None:
    if batch.asset_ids != fitted_asset_ids:
        raise ValueError(
            "prediction asset_ids must exactly match fitted asset_ids in the same order; "
            f"expected {len(fitted_asset_ids)} identifiers, got {len(batch.asset_ids)} "
            "identifiers with different identity or order"
        )


def _validate_context(batch: PortfolioSequenceBatch, config: PortfolioConfig) -> None:
    if config.use_group_embedding and batch.group_ids is None:
        raise ValueError("group_ids are required when use_group_embedding is True")
    if config.use_cost_in_context and batch.costs is None:
        raise ValueError("costs are required when use_cost_in_context is True")
