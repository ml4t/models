"""Persistent-panel PCA baseline."""

from __future__ import annotations

import numpy as np

from ml4t.models.api import PanelBatch
from ml4t.models.configs import PCAConfig
from ml4t.models.latent_factors.base import BaseLatentFactorModel
from ml4t.models.types import FitSummary, LatentFactorState, PersistentPanelBatch


class PCAModel(BaseLatentFactorModel[PCAConfig]):
    """Persistent-panel PCA structural extractor."""

    def __init__(self, config: PCAConfig) -> None:
        super().__init__(config)
        self._asset_mean: np.ndarray | None = None
        self._loadings: np.ndarray | None = None
        self._train_factor_returns: np.ndarray | None = None
        self._asset_ids: tuple[str, ...] = ()

    def fit(self, batch: PanelBatch) -> FitSummary:
        persistent = _require_persistent_panel(batch)
        if persistent.returns is None:
            raise ValueError("PCA requires returns in the training batch")

        returns = np.asarray(persistent.returns, dtype=np.float64)
        asset_mean = np.nanmean(returns, axis=0)
        centered = returns - asset_mean[None, :]
        centered = np.where(np.isfinite(centered), centered, 0.0)

        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        loadings = vt[: self.config.n_factors].T
        factor_returns = centered @ loadings

        self._asset_mean = asset_mean
        self._loadings = loadings
        self._train_factor_returns = factor_returns
        self._asset_ids = persistent.asset_ids
        self._mark_fitted()

        explained_variance = np.var(factor_returns, axis=0, ddof=0).sum()
        total_variance = np.var(centered, axis=0, ddof=0).sum()
        return FitSummary(
            converged=True,
            train_metrics={
                "explained_variance_ratio": float(explained_variance / total_variance)
                if total_variance > 0
                else 0.0,
            },
            notes=("Static loadings extracted from demeaned return panel.",),
        )

    def extract(
        self,
        batch: PanelBatch,
        *,
        checkpoint: int | None = None,
    ) -> LatentFactorState:
        if checkpoint is not None:
            raise ValueError("PCAModel does not expose checkpoints; checkpoint must be None")
        persistent = _require_persistent_panel(batch)
        if not self.is_fitted or self._loadings is None:
            raise RuntimeError("PCA model must be fitted before extract()")

        order = _resolve_asset_order(persistent.asset_ids, self._asset_ids, persistent.n_assets)
        loadings = self._loadings[order]
        n_periods = persistent.n_periods
        asset_betas = np.broadcast_to(
            loadings[None, :, :],
            (n_periods, loadings.shape[0], loadings.shape[1]),
        ).copy()
        factor_returns = None
        if persistent.returns is not None and self._asset_mean is not None:
            asset_mean = self._asset_mean[order]
            centered = np.asarray(persistent.returns, dtype=np.float64) - asset_mean[None, :]
            centered = np.where(np.isfinite(centered), centered, 0.0)
            factor_returns = centered @ loadings

        return LatentFactorState(
            asset_betas=asset_betas,
            factor_returns=factor_returns,
            checkpoint_epoch=None,
            timestamps=persistent.timestamps,
            asset_ids=persistent.asset_ids or self._asset_ids,
            metadata={
                "model_name": self.config.model_name,
                "persistent_entities": True,
            },
        )


def _require_persistent_panel(batch: PanelBatch) -> PersistentPanelBatch:
    if not isinstance(batch, PersistentPanelBatch):
        raise TypeError("PCA requires PersistentPanelBatch input")
    return batch


def _resolve_asset_order(
    requested: tuple[str, ...],
    fitted: tuple[str, ...],
    n_assets: int,
) -> np.ndarray:
    if not fitted or not requested:
        return np.arange(len(fitted) if fitted else n_assets, dtype=np.int64)
    if set(requested) != set(fitted):
        raise ValueError(
            f"prediction asset_ids must match fitted asset_ids; expected {fitted}, got {requested}"
        )
    fitted_index = {asset: index for index, asset in enumerate(fitted)}
    return np.asarray([fitted_index[asset] for asset in requested], dtype=np.int64)
