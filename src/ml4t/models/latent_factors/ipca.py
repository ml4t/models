"""Instrumented PCA on dated cross-sections."""

from __future__ import annotations

import numpy as np

from ml4t.models.api import PanelBatch
from ml4t.models.configs import IPCAConfig
from ml4t.models.latent_factors.base import BaseLatentFactorModel
from ml4t.models.types import CrossSectionBatch, FitSummary, LatentFactorState


class IPCAModel(BaseLatentFactorModel[IPCAConfig]):
    """Instrumented PCA structural extractor for ragged cross-sections."""

    def __init__(self, config: IPCAConfig) -> None:
        super().__init__(config)
        self._gamma: np.ndarray | None = None
        self._train_factor_returns: np.ndarray | None = None
        self._asset_ids: tuple[str, ...] = ()
        self._n_features: int | None = None
        self._fit_iterations = 0
        self._fit_converged = False
        self._fit_parameter_delta = float("inf")
        self._fit_objective_delta = float("inf")
        self._fit_forecast_delta = float("inf")

    @property
    def gamma(self) -> np.ndarray:
        if self._gamma is None:
            raise RuntimeError("IPCA model must be fitted before gamma is available")
        return self._gamma

    @property
    def train_factor_returns(self) -> np.ndarray:
        if self._train_factor_returns is None:
            raise RuntimeError("IPCA model must be fitted before train_factor_returns is available")
        return self._train_factor_returns

    def fit(self, batch: PanelBatch) -> FitSummary:
        cross_section = _require_cross_section(batch)
        if cross_section.returns is None:
            raise ValueError("IPCA requires returns in the training batch")

        characteristics = np.asarray(cross_section.characteristics, dtype=np.float64)
        returns = np.asarray(cross_section.returns, dtype=np.float64)
        mask = _resolve_mask(cross_section)

        train_designs, train_targets = _extract_cross_sections(
            characteristics=characteristics,
            returns=returns,
            mask=mask,
        )
        if not train_designs:
            raise ValueError("IPCA received no valid training cross-sections")

        n_instruments = train_designs[0].shape[1]
        if self.config.n_factors < 1 or self.config.n_factors > n_instruments:
            raise ValueError(
                f"n_factors must be in [1, {n_instruments}]; got {self.config.n_factors}"
            )

        train_ztz = np.stack([design_t.T @ design_t for design_t in train_designs])
        train_zty = np.stack(
            [
                design_t.T @ target_t
                for design_t, target_t in zip(
                    train_designs,
                    train_targets,
                    strict=True,
                )
            ]
        )

        gamma = _initialize_gamma(
            designs=train_designs,
            targets=train_targets,
            n_factors=self.config.n_factors,
        )
        factor_history = np.zeros((len(train_designs), self.config.n_factors), dtype=np.float64)
        converged = False
        previous_objective = float("inf")
        target_sum_squares = float(sum(target @ target for target in train_targets))

        for iteration in range(1, self.config.max_iter + 1):
            previous_gamma = gamma.copy()
            previous_factors = factor_history.copy()

            factor_history = _estimate_factors(
                train_ztz=train_ztz,
                train_zty=train_zty,
                gamma=gamma,
                factor_ridge=self.config.factor_ridge,
            )

            gamma = _estimate_gamma(
                train_ztz=train_ztz,
                train_zty=train_zty,
                factor_history=factor_history,
                n_instruments=n_instruments,
                gamma_ridge=self.config.gamma_ridge,
            )

            # ALS parameters are rotationally unidentified. Put every iterate
            # in the same Theta-Y representation before comparing it with the
            # previous iterate; otherwise equivalent rotations can prevent the
            # raw parameter deltas from converging.
            gamma, factor_history = _normalize_theta_y(gamma, factor_history)

            gamma_delta = float(np.max(np.abs(gamma - previous_gamma)))
            factor_delta = float(np.max(np.abs(factor_history - previous_factors)))
            self._fit_parameter_delta = max(gamma_delta, factor_delta)
            previous_forecast = previous_gamma @ previous_factors.mean(axis=0)
            current_forecast = gamma @ factor_history.mean(axis=0)
            self._fit_forecast_delta = float(np.max(np.abs(current_forecast - previous_forecast)))
            objective = _reconstruction_sse(
                train_ztz=train_ztz,
                train_zty=train_zty,
                target_sum_squares=target_sum_squares,
                gamma=gamma,
                factor_history=factor_history,
            )
            if np.isfinite(previous_objective):
                scale = max(abs(previous_objective), np.finfo(np.float64).eps)
                self._fit_objective_delta = abs(objective - previous_objective) / scale
            previous_objective = objective
            if max(self._fit_objective_delta, self._fit_forecast_delta) <= self.config.tol:
                converged = True
                self._fit_iterations = iteration
                break
        else:
            self._fit_iterations = self.config.max_iter

        # Recompute factor history under the final gamma before applying
        # KPS ΘY normalization.
        factor_history = _estimate_factors(
            train_ztz=train_ztz,
            train_zty=train_zty,
            gamma=gamma,
            factor_ridge=self.config.factor_ridge,
        )

        # Apply KPS appendix C.4 ΘY identification: Γ'Γ = I_K and
        # (1/T) Σ_t f_t f_t' diagonal with descending entries.
        gamma, factor_history = _normalize_theta_y(gamma, factor_history)

        self._gamma = gamma
        self._train_factor_returns = factor_history
        self._asset_ids = cross_section.asset_ids
        self._n_features = characteristics.shape[2]
        self._fit_converged = converged
        self._mark_fitted()

        observed_returns = np.concatenate(train_targets, axis=0)
        reconstruction_error = np.concatenate(
            [
                (design_t @ gamma @ factor_t) - target_t
                for design_t, factor_t, target_t in zip(
                    train_designs,
                    factor_history,
                    train_targets,
                    strict=True,
                )
            ]
        )
        mse = float(np.mean(reconstruction_error**2)) if reconstruction_error.size else 0.0

        return FitSummary(
            converged=converged,
            train_metrics={
                "n_train_periods": float(len(train_designs)),
                "n_instruments": float(n_instruments),
                "train_mse": mse,
                "mean_abs_return": float(np.mean(np.abs(observed_returns))),
            },
            notes=("Alternating least squares over instrumented betas and factor returns.",),
        )

    def extract(
        self,
        batch: PanelBatch,
        *,
        checkpoint: int | None = None,
    ) -> LatentFactorState:
        del checkpoint
        cross_section = _require_cross_section(batch)
        if not self.is_fitted or self._gamma is None or self._n_features is None:
            raise RuntimeError("IPCA model must be fitted before extract()")

        characteristics = np.asarray(cross_section.characteristics, dtype=np.float64)
        if characteristics.shape[2] != self._n_features:
            raise ValueError(
                "characteristics feature dimension does not match fitted IPCA model; "
                f"expected {self._n_features}, got {characteristics.shape[2]}"
            )

        mask = _resolve_mask(cross_section)
        asset_betas = np.full(
            (cross_section.n_periods, cross_section.n_assets, self.config.n_factors),
            np.nan,
            dtype=np.float64,
        )
        returns = None
        if cross_section.returns is not None:
            returns = np.asarray(cross_section.returns, dtype=np.float64)
        factor_returns = None
        if returns is not None:
            factor_returns = np.full(
                (cross_section.n_periods, self.config.n_factors),
                np.nan,
                dtype=np.float64,
            )

        for date_idx in range(cross_section.n_periods):
            valid = _valid_rows(
                characteristics[date_idx],
                mask=mask[date_idx],
                returns=None if cross_section.returns is None else cross_section.returns[date_idx],
                require_returns=False,
            )
            if not valid.any():
                continue
            design_t = _augment_chars(characteristics[date_idx, valid])
            betas_t = design_t @ self._gamma
            asset_betas[date_idx, valid] = betas_t

            if factor_returns is not None:
                assert returns is not None
                returns_t = returns[date_idx]
                valid_returns = _valid_rows(
                    characteristics[date_idx],
                    mask=mask[date_idx],
                    returns=returns_t,
                    require_returns=True,
                )
                if valid_returns.any():
                    design_returns_t = _augment_chars(characteristics[date_idx, valid_returns])
                    betas_returns_t = design_returns_t @ self._gamma
                    gram = betas_returns_t.T @ betas_returns_t + self.config.factor_ridge * np.eye(
                        self.config.n_factors,
                        dtype=np.float64,
                    )
                    rhs = betas_returns_t.T @ returns_t[valid_returns].astype(np.float64)
                    factor_returns[date_idx] = _solve_linear_system(gram, rhs)

        return LatentFactorState(
            asset_betas=asset_betas,
            factor_returns=factor_returns,
            checkpoint_epoch=None,
            timestamps=cross_section.timestamps,
            asset_ids=cross_section.asset_ids or self._asset_ids,
            metadata={
                "model_name": self.config.model_name,
                "persistent_entities": False,
                "fit_iterations": self._fit_iterations,
                "fit_converged": self._fit_converged,
                "fit_parameter_delta": self._fit_parameter_delta,
                "fit_objective_delta": self._fit_objective_delta,
                "fit_forecast_delta": self._fit_forecast_delta,
            },
        )


def _require_cross_section(batch: PanelBatch) -> CrossSectionBatch:
    if not isinstance(batch, CrossSectionBatch):
        raise TypeError("IPCA requires CrossSectionBatch input")
    return batch


def _resolve_mask(batch: CrossSectionBatch) -> np.ndarray:
    if batch.mask is None:
        return np.ones((batch.n_periods, batch.n_assets), dtype=bool)
    return np.asarray(batch.mask, dtype=bool)


def _valid_rows(
    characteristics: np.ndarray,
    *,
    mask: np.ndarray,
    returns: np.ndarray | None,
    require_returns: bool,
) -> np.ndarray:
    valid = mask & np.isfinite(characteristics).all(axis=1)
    if require_returns:
        if returns is None:
            raise ValueError("returns are required when require_returns=True")
        valid = valid & np.isfinite(returns)
    return valid


def _extract_cross_sections(
    *,
    characteristics: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    designs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for date_idx in range(characteristics.shape[0]):
        valid = _valid_rows(
            characteristics[date_idx],
            mask=mask[date_idx],
            returns=returns[date_idx],
            require_returns=True,
        )
        if valid.sum() < 2:
            continue
        designs.append(_augment_chars(characteristics[date_idx, valid]))
        targets.append(returns[date_idx, valid].astype(np.float64))
    return designs, targets


def _augment_chars(characteristics: np.ndarray) -> np.ndarray:
    intercept = np.ones((characteristics.shape[0], 1), dtype=np.float64)
    return np.concatenate([intercept, characteristics.astype(np.float64, copy=False)], axis=1)


def _initialize_gamma(
    *,
    designs: list[np.ndarray],
    targets: list[np.ndarray],
    n_factors: int,
) -> np.ndarray:
    moment = np.zeros((designs[0].shape[1], designs[0].shape[1]), dtype=np.float64)
    for design_t, target_t in zip(designs, targets, strict=True):
        cross_moment = design_t.T @ target_t
        moment += np.outer(cross_moment, cross_moment)
    eigvals, eigvecs = np.linalg.eigh(moment)
    order = np.argsort(eigvals)[::-1][:n_factors]
    return eigvecs[:, order]


def _estimate_gamma(
    *,
    train_ztz: np.ndarray,
    train_zty: np.ndarray,
    factor_history: np.ndarray,
    n_instruments: int,
    gamma_ridge: float,
) -> np.ndarray:
    n_factors = factor_history.shape[1]
    ff = np.einsum("tk,tj->tkj", factor_history, factor_history)
    lhs_blocks = np.einsum("tlm,tkj->lkmj", train_ztz, ff)
    rhs_blocks = train_zty.T @ factor_history

    kron_dim = n_instruments * n_factors
    lhs = lhs_blocks.reshape(kron_dim, kron_dim)
    lhs += gamma_ridge * np.eye(kron_dim, dtype=np.float64)
    rhs = rhs_blocks.reshape(kron_dim)

    gamma_vec = _solve_linear_system(lhs, rhs)
    return gamma_vec.reshape(n_instruments, n_factors)


def _estimate_factors(
    *,
    train_ztz: np.ndarray,
    train_zty: np.ndarray,
    gamma: np.ndarray,
    factor_ridge: float,
) -> np.ndarray:
    grams = np.einsum("lk,tlm,mj->tkj", gamma, train_ztz, gamma, optimize=True)
    grams += factor_ridge * np.eye(gamma.shape[1], dtype=np.float64)[None, :, :]
    rhs = train_zty @ gamma
    try:
        return np.linalg.solve(grams, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.stack(
            [_solve_linear_system(gram, target) for gram, target in zip(grams, rhs, strict=True)]
        )


def _reconstruction_sse(
    *,
    train_ztz: np.ndarray,
    train_zty: np.ndarray,
    target_sum_squares: float,
    gamma: np.ndarray,
    factor_history: np.ndarray,
) -> float:
    grams = np.einsum("lk,tlm,mj->tkj", gamma, train_ztz, gamma, optimize=True)
    linear = np.einsum("tl,lk,tk->", train_zty, gamma, factor_history, optimize=True)
    quadratic = np.einsum("tk,tkj,tj->", factor_history, grams, factor_history, optimize=True)
    return float(max(target_sum_squares - 2.0 * linear + quadratic, 0.0))


def _solve_linear_system(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _normalize_theta_y(
    gamma: np.ndarray, factor_history: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate `(gamma, factor_history)` into the KPS ΘY identification set.

    Implements the construction in KPS (2020) appendix C.4: rotate so that
    `Γ' Γ = I_K` (orthonormal loadings) and `(1/T) Σ_t f_t f_t'` is diagonal
    with descending non-negative entries. The rotation is exact:
    `Γ_new · f_new[t] = Γ · f[t]` for every t, so predictions / fit residuals
    are unchanged.
    """
    n_factors = gamma.shape[1]
    if factor_history.shape[1] != n_factors:
        raise ValueError(
            "gamma and factor_history must share the second dimension; got "
            f"gamma={gamma.shape}, factor_history={factor_history.shape}"
        )

    # Step 1: Cholesky orthonormalization of Γ.
    gram_gamma = gamma.T @ gamma
    try:
        chol = np.linalg.cholesky(gram_gamma)
    except np.linalg.LinAlgError:
        # Defensive: rank-deficient Γ. Fall back to symmetric eigendecomp,
        # which yields the same Γ_new'Γ_new = I_K guarantee on the active
        # subspace. Should be unreachable on well-posed inputs.
        eigvals, eigvecs = np.linalg.eigh(gram_gamma)
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        chol = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    chol_inv = np.linalg.inv(chol)  # K × K, lower triangular inverse
    gamma_orthonormal = gamma @ chol_inv.T  # (L, K), Γ'Γ = I_K
    # In column form `f_new = L^T f`; in row form (factor_history shape (T, K))
    # this becomes `f_new = factor_history @ L = factor_history @ chol`.
    f_intermediate = factor_history @ chol  # preserves Γ · f product

    # Step 2: eigendecomp of factor covariance, descending eigenvalues.
    f_cov = (f_intermediate.T @ f_intermediate) / factor_history.shape[0]
    eigvals, eigvecs = np.linalg.eigh(f_cov)
    order = np.argsort(eigvals)[::-1]
    rotation = eigvecs[:, order]  # orthonormal K × K

    # Step 3: deterministic sign convention. Eigenvectors are unique only up
    # to ±. Pin signs so each column of Γ has a non-negative max-magnitude
    # entry, making results reproducible across runs.
    gamma_final = gamma_orthonormal @ rotation
    f_final = f_intermediate @ rotation
    for k in range(n_factors):
        argmax = int(np.argmax(np.abs(gamma_final[:, k])))
        if gamma_final[argmax, k] < 0:
            gamma_final[:, k] *= -1.0
            f_final[:, k] *= -1.0

    return gamma_final, f_final
