"""Risk-premium-aware persistent-panel PCA."""

from __future__ import annotations

import numpy as np

from ml4t.models.api import PanelBatch
from ml4t.models.configs import RPPCAConfig
from ml4t.models.latent_factors.base import BaseLatentFactorModel
from ml4t.models.types import FitSummary, LatentFactorState, PersistentPanelBatch


class RPPCAModel(BaseLatentFactorModel[RPPCAConfig]):
    """Persistent-panel RP-PCA structural extractor."""

    def __init__(self, config: RPPCAConfig) -> None:
        super().__init__(config)
        self._asset_betas: np.ndarray | None = None
        self._factor_weights: np.ndarray | None = None
        self._train_factor_returns: np.ndarray | None = None
        self._eigenvalues: np.ndarray | None = None
        self._asset_ids: tuple[str, ...] = ()

    def fit(self, batch: PanelBatch) -> FitSummary:
        persistent = _require_persistent_panel(batch)
        if persistent.returns is None:
            raise ValueError("RP-PCA requires returns in the training batch")

        returns = np.asarray(persistent.returns, dtype=np.float64)
        returns = np.where(np.isfinite(returns), returns, 0.0)
        n_periods, n_assets = returns.shape
        if self.config.n_factors < 1 or self.config.n_factors > n_assets:
            raise ValueError(
                f"n_factors must be in [1, {n_assets}]; got {self.config.n_factors}"
            )

        normalization = _cross_section_normalization(
            returns,
            scale_by_asset_volatility=self.config.scale_by_asset_volatility,
        )
        weighted_returns = returns @ normalization
        rp_matrix = _risk_premium_matrix(
            weighted_returns,
            gamma=self.config.gamma,
            base_moment=self.config.base_moment,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(rp_matrix)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order][: self.config.n_factors]
        eigenvectors = eigenvectors[:, order][:, : self.config.n_factors]

        loadings = np.linalg.solve(normalization.T, eigenvectors)
        factor_weights = loadings @ np.linalg.inv(loadings.T @ loadings)
        factor_weights = factor_weights @ np.linalg.inv(
            np.sqrt(np.diag(np.diag(factor_weights.T @ factor_weights)))
        )
        factor_returns = returns @ factor_weights
        loadings = _sign_normalize_loadings(loadings, factor_returns)

        if self.config.normalize_loadings == "variance":
            scale = np.sqrt(np.clip(eigenvalues, a_min=1e-12, a_max=None))
            loadings = loadings @ np.diag(scale)
            factor_weights = factor_weights @ np.diag(1.0 / scale)
            factor_returns = returns @ factor_weights
        elif self.config.normalize_loadings != "unit_length":
            raise ValueError(
                "normalize_loadings must be 'unit_length' or 'variance'; "
                f"got {self.config.normalize_loadings!r}"
            )

        if self.config.orthogonalize_factors:
            factor_returns, loadings, factor_weights = _orthogonalize(
                returns=returns,
                factor_returns=factor_returns,
                loadings=loadings,
                factor_weights=factor_weights,
                unit_length=self.config.normalize_loadings == "unit_length",
            )

        asset_betas, alphas, residuals = _time_series_betas(returns, factor_returns)

        self._asset_betas = asset_betas
        self._factor_weights = factor_weights
        self._train_factor_returns = factor_returns
        self._eigenvalues = eigenvalues
        self._asset_ids = persistent.asset_ids
        self._mark_fitted()

        residual_variance = float(np.nanmean(np.var(residuals, axis=0, ddof=0)))
        return FitSummary(
            converged=True,
            train_metrics={
                "mean_abs_factor_premium": float(np.mean(np.abs(np.mean(factor_returns, axis=0)))),
                "mean_factor_sharpe": float(np.mean(np.abs(_factor_sharpes(factor_returns)))),
                "mean_abs_alpha": float(np.mean(np.abs(alphas))),
                "mean_residual_variance": residual_variance,
            },
            notes=("Static RP-PCA loadings estimated from persistent return panels.",),
        )

    def extract(
        self,
        batch: PanelBatch,
        *,
        checkpoint: int | None = None,
    ) -> LatentFactorState:
        del checkpoint
        persistent = _require_persistent_panel(batch)
        if not self.is_fitted or self._asset_betas is None or self._factor_weights is None:
            raise RuntimeError("RP-PCA model must be fitted before extract()")

        factor_returns = None
        if persistent.returns is not None:
            returns = np.asarray(persistent.returns, dtype=np.float64)
            returns = np.where(np.isfinite(returns), returns, 0.0)
            factor_returns = returns @ self._factor_weights

        asset_betas = np.broadcast_to(
            self._asset_betas[None, :, :],
            (persistent.n_periods, self._asset_betas.shape[0], self._asset_betas.shape[1]),
        ).copy()
        metadata: dict[str, object] = {
            "model_name": self.config.model_name,
            "persistent_entities": True,
            "gamma": self.config.gamma,
            "base_moment": self.config.base_moment,
        }
        if self._eigenvalues is not None:
            metadata["eigenvalues"] = tuple(float(value) for value in self._eigenvalues)

        return LatentFactorState(
            asset_betas=asset_betas,
            factor_returns=factor_returns,
            timestamps=persistent.timestamps,
            asset_ids=persistent.asset_ids or self._asset_ids,
            metadata=metadata,
        )


def _require_persistent_panel(batch: PanelBatch) -> PersistentPanelBatch:
    if not isinstance(batch, PersistentPanelBatch):
        raise TypeError("RP-PCA requires PersistentPanelBatch input")
    return batch


def _cross_section_normalization(
    returns: np.ndarray,
    *,
    scale_by_asset_volatility: bool,
) -> np.ndarray:
    if not scale_by_asset_volatility:
        return np.eye(returns.shape[1], dtype=np.float64)
    centered = returns - np.mean(returns, axis=0, keepdims=True)
    variance = np.mean(centered**2, axis=0)
    scale = np.sqrt(np.clip(variance, a_min=1e-12, a_max=None))
    return np.diag(1.0 / scale)


def _risk_premium_matrix(
    returns: np.ndarray,
    *,
    gamma: float,
    base_moment: str,
) -> np.ndarray:
    mean_returns = np.mean(returns, axis=0)
    if base_moment == "covariance":
        centered = returns - mean_returns[None, :]
        base = centered.T @ centered / max(returns.shape[0] - 1, 1)
    elif base_moment == "second_moment":
        base = returns.T @ returns / max(returns.shape[0], 1)
    else:
        raise ValueError(
            "base_moment must be 'covariance' or 'second_moment'; "
            f"got {base_moment!r}"
        )
    return base + gamma * np.outer(mean_returns, mean_returns)


def _sign_normalize_loadings(loadings: np.ndarray, factor_returns: np.ndarray) -> np.ndarray:
    signs = np.sign(np.mean(factor_returns, axis=0))
    signs = np.where(signs == 0.0, 1.0, signs)
    return loadings @ np.diag(signs)


def _orthogonalize(
    *,
    returns: np.ndarray,
    factor_returns: np.ndarray,
    loadings: np.ndarray,
    factor_weights: np.ndarray,
    unit_length: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_periods = factor_returns.shape[0]
    centered = factor_returns - np.mean(factor_returns, axis=0, keepdims=True)
    _, r = np.linalg.qr(centered / np.sqrt(max(n_periods, 1)))
    rotation = np.linalg.inv(r) @ np.diag(np.diag(r)) if unit_length else np.linalg.inv(r)
    factor_weights = factor_weights @ rotation
    if unit_length:
        scale = np.sqrt(np.clip(np.diag(factor_weights.T @ factor_weights), a_min=1e-12, a_max=None))
        factor_weights = factor_weights @ np.diag(1.0 / scale)
    factor_returns = returns @ factor_weights
    signs = np.sign(np.mean(factor_returns, axis=0))
    signs = np.where(signs == 0.0, 1.0, signs)
    sign_matrix = np.diag(signs)
    factor_returns = factor_returns @ sign_matrix
    factor_weights = factor_weights @ sign_matrix
    loadings = loadings @ np.linalg.inv(rotation) @ sign_matrix
    return factor_returns, loadings, factor_weights


def _time_series_betas(
    returns: np.ndarray,
    factor_returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(returns.shape[0], dtype=np.float64), factor_returns])
    coeffs, *_ = np.linalg.lstsq(design, returns, rcond=None)
    fitted = design @ coeffs
    residuals = returns - fitted
    return coeffs[1:].T, coeffs[0], residuals


def _factor_sharpes(factor_returns: np.ndarray) -> np.ndarray:
    mean = np.mean(factor_returns, axis=0)
    std = np.std(factor_returns, axis=0, ddof=0)
    return np.divide(mean, std, out=np.zeros_like(mean), where=std > 1e-12)
