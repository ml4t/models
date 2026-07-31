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
    """Project each date onto joint gross, net, and per-asset constraints."""

    if gross_exposure < abs(net_exposure):
        raise ValueError(
            "infeasible portfolio constraints: gross_exposure must be at least "
            f"abs(net_exposure); got gross={gross_exposure}, net={net_exposure}"
        )
    if max_abs_weight is not None and max_abs_weight <= 0.0:
        raise ValueError(
            "infeasible portfolio constraints: max_abs_weight must be positive; "
            f"got {max_abs_weight}"
        )

    normalized = np.zeros_like(weights, dtype=np.float64)
    batch_size, n_periods, _ = weights.shape

    for batch_idx in range(batch_size):
        for period_idx in range(n_periods):
            valid = mask[batch_idx, period_idx]
            if not valid.any():
                continue
            row = np.zeros(weights.shape[2], dtype=np.float64)
            row_valid = np.asarray(weights[batch_idx, period_idx, valid], dtype=np.float64)
            if not np.isfinite(row_valid).all():
                raise ValueError("portfolio weights must be finite at available positions")
            row[valid] = _project_weight_row(
                row_valid,
                gross_exposure=gross_exposure,
                net_exposure=net_exposure,
                max_abs_weight=max_abs_weight,
            )
            normalized[batch_idx, period_idx] = row

    return normalized


def _project_weight_row(
    values: np.ndarray,
    *,
    gross_exposure: float,
    net_exposure: float,
    max_abs_weight: float | None,
) -> np.ndarray:
    long_total = 0.5 * (gross_exposure + net_exposure)
    short_total = 0.5 * (gross_exposure - net_exposure)
    if max_abs_weight is None:
        return _normalize_uncapped_weight_row(values, long_total, short_total)
    cap = max_abs_weight
    order = np.argsort(values, kind="stable")
    best: np.ndarray | None = None
    best_error = float("inf")
    tolerance = 1e-12

    for split in range(values.size + 1):
        short_indices = order[:split]
        long_indices = order[split:]
        if short_total > cap * short_indices.size + tolerance:
            continue
        if long_total > cap * long_indices.size + tolerance:
            continue
        candidate = np.zeros_like(values)
        candidate[short_indices] = -_project_capped_simplex(
            -values[short_indices], short_total, cap
        )
        candidate[long_indices] = _project_capped_simplex(values[long_indices], long_total, cap)
        error = float(np.sum((candidate - values) ** 2))
        if error < best_error:
            best = candidate
            best_error = error

    if best is None:
        raise ValueError(
            "infeasible portfolio constraints: available assets and max_abs_weight "
            f"cannot support gross={gross_exposure}, net={net_exposure}, cap={max_abs_weight}"
        )
    return best


def _normalize_uncapped_weight_row(
    values: np.ndarray,
    long_total: float,
    short_total: float,
) -> np.ndarray:
    if values.size < int(long_total > 0.0) + int(short_total > 0.0):
        raise ValueError(
            "infeasible portfolio constraints: available assets cannot support "
            f"long={long_total} and short={short_total}"
        )
    centered = values - np.mean(values)
    positive = np.maximum(centered, 0.0)
    negative = np.maximum(-centered, 0.0)
    positive_total = float(positive.sum())
    negative_total = float(negative.sum())
    if positive_total > 0.0 and negative_total > 0.0:
        return positive * (long_total / positive_total) - negative * (short_total / negative_total)

    normalized = np.zeros_like(values)
    if long_total == 0.0:
        normalized.fill(-short_total / values.size)
        return normalized
    if short_total == 0.0:
        normalized.fill(long_total / values.size)
        return normalized

    long_count = int(round(values.size * long_total / (long_total + short_total)))
    long_count = min(max(long_count, 1), values.size - 1)
    order = np.argsort(values, kind="stable")
    normalized[order[:-long_count]] = -short_total / (values.size - long_count)
    normalized[order[-long_count:]] = long_total / long_count
    return normalized


def _project_capped_simplex(values: np.ndarray, total: float, cap: float) -> np.ndarray:
    if total <= 0.0:
        return np.zeros_like(values)
    if cap >= total:
        return _project_simplex(values, total)
    lower = float(np.min(values - cap))
    upper = float(np.max(values))
    for _ in range(100):
        threshold = 0.5 * (lower + upper)
        projected = np.clip(values - threshold, 0.0, cap)
        if projected.sum() > total:
            lower = threshold
        else:
            upper = threshold
    projected = np.clip(values - upper, 0.0, cap)
    residual = total - float(projected.sum())
    if residual > 0.0:
        for index in np.flatnonzero(projected < cap):
            addition = min(residual, cap - projected[index])
            projected[index] += addition
            residual -= addition
            if residual <= 1e-14:
                break
    return projected


def _project_simplex(values: np.ndarray, total: float) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(values)
    descending = np.sort(values)[::-1]
    cumulative = np.cumsum(descending) - total
    candidates = descending - cumulative / np.arange(1, values.size + 1) > 0.0
    active = int(np.flatnonzero(candidates)[-1])
    threshold = cumulative[active] / (active + 1)
    return np.maximum(values - threshold, 0.0)


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
