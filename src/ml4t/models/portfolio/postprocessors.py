"""Portfolio-weight post-processing hooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml4t.models.types import PortfolioSequenceBatch, PortfolioWeightsResult


@dataclass(frozen=True, slots=True)
class WeightConstraintPostprocessor:
    """Apply exposure, clipping, and turnover constraints to portfolio weights."""

    gross_exposure: float = 1.0
    net_exposure: float = 0.0
    max_abs_weight: float | None = None
    turnover_limit: float | None = None

    def transform(
        self,
        batch: PortfolioSequenceBatch,
        weights: PortfolioWeightsResult,
    ) -> PortfolioWeightsResult:
        mask = (
            np.asarray(batch.mask, dtype=bool)
            if batch.mask is not None
            else np.ones(weights.weights.shape, dtype=bool)
        )
        constrained = normalize_cross_sectional_weights(
            weights.weights,
            mask=mask,
            gross_exposure=self.gross_exposure,
            net_exposure=self.net_exposure,
            max_abs_weight=self.max_abs_weight,
        )
        if self.turnover_limit is not None:
            constrained = apply_turnover_limit(
                constrained,
                previous_weights=batch.prev_weights,
                mask=mask,
                turnover_limit=self.turnover_limit,
            )
        return PortfolioWeightsResult(
            weights=constrained,
            checkpoint_step=weights.checkpoint_step,
            timestamps=weights.timestamps,
            asset_ids=weights.asset_ids,
            metadata={**weights.metadata, "postprocessor": "weight_constraints"},
        )


def normalize_cross_sectional_weights(
    weights: np.ndarray,
    *,
    mask: np.ndarray,
    gross_exposure: float,
    net_exposure: float,
    max_abs_weight: float | None,
) -> np.ndarray:
    """Normalize cross-sectional weights date by date."""

    normalized = np.zeros_like(weights, dtype=np.float64)
    batch_size, n_periods, _ = weights.shape

    for batch_idx in range(batch_size):
        for period_idx in range(n_periods):
            valid = mask[batch_idx, period_idx]
            if not valid.any():
                continue
            row = np.asarray(weights[batch_idx, period_idx], dtype=np.float64).copy()
            row[~valid] = 0.0
            row_valid = row[valid]
            row_valid = row_valid - row_valid.mean()
            row_valid = row_valid + (net_exposure / max(int(valid.sum()), 1))
            gross = np.abs(row_valid).sum()
            if gross > 0:
                row_valid = row_valid * (gross_exposure / gross)
            if max_abs_weight is not None:
                row_valid = np.clip(row_valid, -max_abs_weight, max_abs_weight)
            row[:] = 0.0
            row[valid] = row_valid
            normalized[batch_idx, period_idx] = row

    return normalized


def apply_turnover_limit(
    weights: np.ndarray,
    *,
    previous_weights: np.ndarray | None,
    mask: np.ndarray,
    turnover_limit: float,
) -> np.ndarray:
    """Scale cross-sectional weight changes to satisfy an L1 turnover cap."""

    constrained = np.asarray(weights, dtype=np.float64).copy()
    batch_size, n_periods, _ = constrained.shape
    if turnover_limit <= 0:
        return np.zeros_like(constrained)

    previous = (
        np.asarray(previous_weights, dtype=np.float64).copy()
        if previous_weights is not None
        else np.zeros((batch_size, constrained.shape[2]), dtype=np.float64)
    )

    for batch_idx in range(batch_size):
        current_prev = previous[batch_idx]
        for period_idx in range(n_periods):
            valid = mask[batch_idx, period_idx]
            target = constrained[batch_idx, period_idx].copy()
            target[~valid] = 0.0
            current_prev = current_prev.copy()
            current_prev[~valid] = 0.0
            turnover = np.abs(target - current_prev).sum()
            if turnover > turnover_limit:
                scale = turnover_limit / turnover
                target = current_prev + scale * (target - current_prev)
            constrained[batch_idx, period_idx] = target
            current_prev = target
    return constrained
