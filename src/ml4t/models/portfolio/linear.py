"""Deterministic linear feature portfolio baseline."""

from __future__ import annotations

import numpy as np

from ml4t.models.configs import LinearPortfolioConfig
from ml4t.models.portfolio.base import BasePortfolioModel
from ml4t.models.portfolio.postprocessors import normalize_cross_sectional_weights
from ml4t.models.portfolio.runtime import validate_portfolio_batch
from ml4t.models.types import FitSummary, PortfolioSequenceBatch, PortfolioWeightsResult


class LinearFeaturePortfolioModel(BasePortfolioModel):
    """Fit pooled linear factor scores and map them to cross-sectional portfolio weights."""

    def __init__(self, config: LinearPortfolioConfig) -> None:
        super().__init__(config)
        self.config: LinearPortfolioConfig = config
        self._coefficients: np.ndarray | None = None
        self._asset_ids: tuple[str, ...] = ()
        self._n_features: int | None = None

    @property
    def available_checkpoints(self) -> tuple[int, ...]:
        return (1,) if self.is_fitted else ()

    def fit(
        self,
        batch: PortfolioSequenceBatch,
        *,
        validation_batch: PortfolioSequenceBatch | None = None,
    ) -> FitSummary:
        validate_portfolio_batch(batch)
        del validation_batch

        features = np.asarray(batch.features, dtype=np.float64)
        returns = np.asarray(batch.returns, dtype=np.float64)
        mask = (
            np.asarray(batch.mask, dtype=bool)
            if batch.mask is not None
            else np.ones(features.shape[:3], dtype=bool)
        )
        valid = mask & np.isfinite(returns) & np.isfinite(features).all(axis=-1)
        if not valid.any():
            raise ValueError("LinearFeaturePortfolioModel received no valid training observations")

        design = features[valid]
        target = returns[valid]
        if self.config.fit_intercept:
            design = np.column_stack([np.ones(design.shape[0], dtype=np.float64), design])

        ridge_penalty = self.config.ridge_alpha * np.eye(design.shape[1], dtype=np.float64)
        if self.config.fit_intercept:
            ridge_penalty[0, 0] = 0.0
        lhs = design.T @ design + ridge_penalty
        rhs = design.T @ target
        coefficients = np.linalg.solve(lhs, rhs)

        self._coefficients = coefficients.astype(np.float64)
        self._asset_ids = batch.asset_ids
        self._n_features = features.shape[3]
        self._mark_fitted()

        predictions = design @ coefficients
        residual = target - predictions
        return FitSummary(
            converged=True,
            train_metrics={
                "n_train_obs": float(target.shape[0]),
                "train_rmse": float(np.sqrt(np.mean(residual**2))),
            },
            best_epoch=1,
            notes=("Pooled linear feature model mapped to portfolio weights.",),
        )

    def predict(
        self,
        batch: PortfolioSequenceBatch,
        *,
        checkpoint: int | None = None,
    ) -> PortfolioWeightsResult:
        if not self.is_fitted or self._coefficients is None or self._n_features is None:
            raise RuntimeError("LinearFeaturePortfolioModel must be fitted before predict()")
        if checkpoint is not None and checkpoint != 1:
            raise ValueError("LinearFeaturePortfolioModel only exposes checkpoint=1")
        if batch.features.shape[3] != self._n_features:
            raise ValueError("prediction batch feature dimension does not match the fitted model")

        features = np.asarray(batch.features, dtype=np.float64)
        mask = (
            np.asarray(batch.mask, dtype=bool)
            if batch.mask is not None
            else np.ones(features.shape[:3], dtype=bool)
        )
        scores = np.einsum("btnf,f->btn", features, self._feature_coefficients, optimize=True)
        if self.config.fit_intercept:
            scores = scores + self._intercept
        scores = np.where(mask, scores, 0.0)
        diagonal_weights = normalize_cross_sectional_weights(
            scores,
            mask=mask,
            gross_exposure=self.config.gross_exposure,
            net_exposure=self.config.net_exposure,
            max_abs_weight=self.config.max_abs_weight,
        )

        return PortfolioWeightsResult(
            weights=diagonal_weights,
            checkpoint_step=1,
            timestamps=batch.timestamps,
            asset_ids=batch.asset_ids or self._asset_ids,
            metadata={"model_name": self.config.model_name},
        )

    @property
    def _intercept(self) -> float:
        assert self._coefficients is not None
        return float(self._coefficients[0]) if self.config.fit_intercept else 0.0

    @property
    def _feature_coefficients(self) -> np.ndarray:
        assert self._coefficients is not None
        return (
            self._coefficients[1:].astype(np.float64, copy=False)
            if self.config.fit_intercept
            else self._coefficients.astype(np.float64, copy=False)
        )
