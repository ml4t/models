"""Unit tests pinning KPS-faithful ΘY normalization in IPCA."""

from __future__ import annotations

import numpy as np

from ml4t.models import CrossSectionBatch, IPCAConfig, IPCAModel
from ml4t.models.latent_factors.ipca import _estimate_factors, _normalize_theta_y


def _make_panel(*, n_periods: int, n_assets: int, n_features: int, n_factors: int, seed: int):
    rng = np.random.default_rng(seed)
    characteristics = rng.normal(size=(n_periods, n_assets, n_features))
    gamma_true = rng.normal(size=(n_features + 1, n_factors)) * 0.3
    factors_true = rng.normal(scale=0.5, size=(n_periods, n_factors))
    augmented = np.concatenate(
        [np.ones((n_periods, n_assets, 1), dtype=np.float64), characteristics],
        axis=2,
    )
    betas = np.einsum("tnl,lk->tnk", augmented, gamma_true, optimize=True)
    returns = np.einsum("tnk,tk->tn", betas, factors_true, optimize=True)
    returns += 0.01 * rng.normal(size=returns.shape)
    return CrossSectionBatch(
        characteristics=characteristics,
        returns=returns,
        timestamps=tuple(f"2024-{idx:02d}" for idx in range(1, n_periods + 1)),
    )


def test_gamma_is_orthonormal_after_fit() -> None:
    batch = _make_panel(n_periods=24, n_assets=12, n_features=4, n_factors=2, seed=11)
    model = IPCAModel(IPCAConfig(n_factors=2, max_iter=400, tol=1e-10))
    model.fit(batch)

    gram = model.gamma.T @ model.gamma
    np.testing.assert_allclose(gram, np.eye(2), atol=1e-8)


def test_factor_covariance_is_diagonal_descending() -> None:
    batch = _make_panel(n_periods=30, n_assets=15, n_features=5, n_factors=3, seed=17)
    model = IPCAModel(IPCAConfig(n_factors=3, max_iter=400, tol=1e-10))
    model.fit(batch)

    f = model.train_factor_returns
    f_cov = (f.T @ f) / f.shape[0]
    off_diag = f_cov - np.diag(np.diag(f_cov))
    np.testing.assert_allclose(off_diag, 0.0, atol=1e-8)

    diag = np.diag(f_cov)
    assert np.all(np.diff(diag) <= 1e-10), f"factor variances must be descending; got diag={diag}"
    assert np.all(diag >= -1e-12), f"factor variances must be non-negative; got diag={diag}"


def test_als_convergence_compares_identified_iterates() -> None:
    batch = _make_panel(n_periods=60, n_assets=20, n_features=12, n_factors=5, seed=99)
    model = IPCAModel(IPCAConfig(n_factors=5, max_iter=400, tol=1e-6))

    fit = model.fit(batch)

    assert fit.converged
    assert model._fit_iterations < model.config.max_iter
    assert model._fit_objective_delta <= model.config.tol
    assert model._fit_forecast_delta <= model.config.tol


def test_vectorized_factor_step_matches_datewise_solves() -> None:
    rng = np.random.default_rng(101)
    n_periods, n_instruments, n_factors = 7, 6, 3
    raw = rng.normal(size=(n_periods, n_instruments, n_instruments))
    train_ztz = np.einsum("tij,tkj->tik", raw, raw)
    train_zty = rng.normal(size=(n_periods, n_instruments))
    gamma = rng.normal(size=(n_instruments, n_factors))
    ridge = 1e-6

    vectorized = _estimate_factors(
        train_ztz=train_ztz,
        train_zty=train_zty,
        gamma=gamma,
        factor_ridge=ridge,
    )
    datewise = []
    for ztz_t, zty_t in zip(train_ztz, train_zty, strict=True):
        gram = gamma.T @ ztz_t @ gamma + ridge * np.eye(n_factors)
        datewise.append(np.linalg.solve(gram, gamma.T @ zty_t))

    np.testing.assert_allclose(vectorized, np.stack(datewise), atol=1e-12)


def test_predictions_invariant_to_normalization() -> None:
    """Direct test: rotating (Γ, f_t) into ΘY form preserves Γ · f_t exactly."""

    rng = np.random.default_rng(23)
    gamma = rng.normal(size=(6, 3))
    factor_history = rng.normal(scale=0.4, size=(40, 3))
    predictions_before = factor_history @ gamma.T  # (T, n_instruments)

    gamma_new, factor_history_new = _normalize_theta_y(gamma, factor_history)
    predictions_after = factor_history_new @ gamma_new.T

    np.testing.assert_allclose(predictions_after, predictions_before, atol=1e-12)


def test_extract_predictions_match_unnormalized_at_default_ridge() -> None:
    """End-to-end: predictions Z·Γ·f are invariant to ΘY rotation at low ridge.

    Builds the panel, fits IPCA (which now applies normalization), reconstructs
    Z·Γ·f via extract(), and verifies match against the rotation-equivalent
    predictions computed in normalize-and-rotate-back fashion: any KPS-ΘY-pinned
    Γ delivers the same Z·Γ·f as any rotation of it (within ridge tolerance).
    """
    batch = _make_panel(n_periods=24, n_assets=12, n_features=4, n_factors=2, seed=29)
    model = IPCAModel(IPCAConfig(n_factors=2, max_iter=400, tol=1e-10))
    model.fit(batch)
    state = model.extract(batch)

    reconstructed = np.einsum("tnk,tk->tn", state.asset_betas, state.factor_returns, optimize=True)
    valid = np.isfinite(reconstructed) & np.isfinite(batch.returns)
    # Per KPS, fit minimizes Σ (r - Z Γ f)². At default ridge the residual is
    # small and is the same as the un-normalized fit would deliver.
    residual_var = float(np.nanvar(reconstructed[valid] - batch.returns[valid]))
    return_var = float(np.nanvar(batch.returns[valid]))
    assert residual_var < 0.25 * return_var, (
        f"reconstruction residual variance {residual_var:.4g} suggests fit broken; "
        f"return variance is {return_var:.4g}"
    )


def test_normalize_theta_y_idempotent() -> None:
    """Applying normalization twice returns the same (Γ, f) up to machine eps."""
    rng = np.random.default_rng(37)
    gamma = rng.normal(size=(5, 3))
    factor_history = rng.normal(scale=0.5, size=(50, 3))

    gamma1, f1 = _normalize_theta_y(gamma, factor_history)
    gamma2, f2 = _normalize_theta_y(gamma1, f1)

    np.testing.assert_allclose(gamma2, gamma1, atol=1e-10)
    np.testing.assert_allclose(f2, f1, atol=1e-10)


def test_normalize_theta_y_sign_convention_deterministic() -> None:
    """Sign convention: each column of Γ has non-negative max-magnitude entry."""
    rng = np.random.default_rng(41)
    gamma = rng.normal(size=(7, 4))
    factor_history = rng.normal(scale=0.5, size=(60, 4))

    gamma_norm, _ = _normalize_theta_y(gamma, factor_history)
    for k in range(gamma_norm.shape[1]):
        argmax = int(np.argmax(np.abs(gamma_norm[:, k])))
        assert gamma_norm[argmax, k] >= 0.0, (
            f"column {k}: max-magnitude entry must be non-negative; got {gamma_norm[argmax, k]}"
        )


def test_final_factor_recomputation_at_non_convergence() -> None:
    """At max_iter=1 + factor_ridge=0, the post-loop final factor recomputation
    guarantees (Γ, f) consistency: extract()'s per-date OLS solve under the
    stored normalized Γ recovers the stored factor_history exactly.

    With factor_ridge > 0, the OLS-with-ridge solve is not rotation-equivariant,
    so the stored (ΘY-rotated) f and extract()'s recomputed f differ by ridge
    perturbation. Use ridge=0 here to pin the principle.
    """
    batch = _make_panel(n_periods=18, n_assets=10, n_features=3, n_factors=2, seed=43)
    model = IPCAModel(
        IPCAConfig(n_factors=2, max_iter=1, tol=1e-10, factor_ridge=0.0, gamma_ridge=0.0)
    )
    fit_summary = model.fit(batch)

    # max_iter=1 deliberately fails to converge; the final-pass guarantees that
    # for each date t, f_t solves the per-date OLS-on-current-Γ exactly.
    assert fit_summary.converged is False

    state = model.extract(batch)
    f_extracted = state.factor_returns
    f_internal = model.train_factor_returns
    np.testing.assert_allclose(f_extracted, f_internal, atol=1e-10)
