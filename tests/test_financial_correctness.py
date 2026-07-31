from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from ml4t.models import PortfolioConfig, PortfolioSequenceBatch, RPPCAConfig
from ml4t.models._internal.latent_factor_utils import compute_managed_portfolios
from ml4t.models._internal.stochastic_discount_factor_nn import (
    conditional_loss,
    construct_stochastic_discount_factor,
    unconditional_loss,
)
from ml4t.models.latent_factors.ipca import _normalize_theta_y
from ml4t.models.latent_factors.rp_pca import _risk_premium_matrix
from ml4t.models.portfolio import runtime as portfolio_runtime
from ml4t.models.portfolio.losses import compute_net_portfolio_returns, softmin_sharpe
from ml4t.models.portfolio.postprocessors import normalize_cross_sectional_weights


def test_rp_pca_default_matches_lettau_pelger_signal_matrix() -> None:
    returns = np.array(
        [
            [-2.0, 3.0],
            [2.0, 3.0],
            [-2.0, 3.0],
            [2.0, 3.0],
        ]
    )
    second_moment = returns.T @ returns / returns.shape[0]
    mean = returns.mean(axis=0)

    assert RPPCAConfig().base_moment == "second_moment"
    np.testing.assert_allclose(
        _risk_premium_matrix(returns, gamma=0.0, base_moment="second_moment"),
        second_moment,
    )
    np.testing.assert_allclose(
        _risk_premium_matrix(returns, gamma=-1.0, base_moment="second_moment"),
        second_moment - np.outer(mean, mean),
    )


def test_cae_managed_portfolios_match_joint_cross_sectional_ols() -> None:
    characteristics = np.array(
        [
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )
    design = np.column_stack(
        [characteristics[0], np.ones(characteristics.shape[1], dtype=np.float64)]
    )
    coefficients = np.array([2.0, -1.0, 0.5])
    returns = (design @ coefficients)[None, :]

    actual = compute_managed_portfolios(characteristics, returns)[0, 0]

    np.testing.assert_allclose(actual, coefficients, atol=1e-6)


def test_cae_managed_portfolios_use_minimum_norm_for_collinear_design() -> None:
    characteristics = np.array([[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]])
    returns = np.array([[1.0, np.nan, 3.0]])
    design = np.column_stack([characteristics[0, [0, 2]], np.ones(2)])
    expected, *_ = np.linalg.lstsq(design, returns[0, [0, 2]], rcond=None)

    actual = compute_managed_portfolios(characteristics, returns)[0, 0]

    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_ipca_identification_makes_factor_means_nonnegative() -> None:
    gamma = np.array([[2.0], [1.0]])
    factor_history = np.array([[-1.0], [-2.0], [-3.0]])

    gamma_normalized, factors_normalized = _normalize_theta_y(gamma, factor_history)

    assert factors_normalized.mean(axis=0).item() >= 0.0
    np.testing.assert_allclose(
        factors_normalized @ gamma_normalized.T,
        factor_history @ gamma.T,
        atol=1e-12,
    )


def test_sdf_uses_paper_weight_sign() -> None:
    returns = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    mask = torch.ones((2, 2), dtype=torch.bool)

    actual = construct_stochastic_discount_factor(returns, weights, mask)
    expected = 1.0 - (returns * weights.reshape(2, 2)).sum(dim=1)

    torch.testing.assert_close(actual, expected)


def test_sdf_losses_ignore_nonfinite_values_outside_mask() -> None:
    returns = torch.tensor([[0.1, float("nan")], [0.2, float("inf")]], dtype=torch.float64)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    mask = torch.tensor([[True, False], [True, False]])
    counts = mask.float().sum(dim=0)
    instruments = torch.tensor(
        [[[1.0, float("nan")], [1.0, float("inf")]]],
        dtype=torch.float64,
    )
    clean_returns = torch.where(mask, returns, torch.zeros_like(returns))
    clean_instruments = torch.where(mask.unsqueeze(0), instruments, torch.zeros_like(instruments))

    actual_unconditional, _ = unconditional_loss(weights, returns, mask, counts)
    expected_unconditional, _ = unconditional_loss(weights, clean_returns, mask, counts)
    actual_conditional, _ = conditional_loss(weights, instruments, returns, mask, counts)
    expected_conditional, _ = conditional_loss(
        weights, clean_instruments, clean_returns, mask, counts
    )

    torch.testing.assert_close(actual_unconditional, expected_unconditional)
    torch.testing.assert_close(actual_conditional, expected_conditional)
    assert torch.isfinite(actual_unconditional)
    assert torch.isfinite(actual_conditional)


def test_portfolio_returns_ignore_nonfinite_values_outside_mask() -> None:
    weights = torch.ones((1, 2, 2), dtype=torch.float64)
    returns = torch.tensor(
        [[[0.01, float("nan")], [0.02, 0.03]]],
        dtype=torch.float64,
    )
    vol_scale = torch.tensor(
        [[[1.0, float("inf")], [1.0, 1.0]]],
        dtype=torch.float64,
    )
    mask = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]], dtype=torch.float64)
    clean_returns = torch.where(mask.bool(), returns, torch.zeros_like(returns))
    clean_vol_scale = torch.where(mask.bool(), vol_scale, torch.zeros_like(vol_scale))

    actual = compute_net_portfolio_returns(
        weights=weights,
        forward_returns=returns,
        vol_scale=vol_scale,
        mask=mask,
        costs=torch.ones(2, dtype=torch.float64),
        gamma_cost=0.5,
    )
    expected = compute_net_portfolio_returns(
        weights=weights,
        forward_returns=clean_returns,
        vol_scale=clean_vol_scale,
        mask=mask,
        costs=torch.ones(2, dtype=torch.float64),
        gamma_cost=0.5,
    )

    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()


def test_softmin_sharpe_is_stable_for_extreme_finite_values() -> None:
    for values, tau in (
        (torch.tensor([-1000.0, 1.0], dtype=torch.float64), 0.01),
        (torch.tensor([30.0, 31.0], dtype=torch.float32), 0.2),
    ):
        actual = softmin_sharpe(values, tau=tau)
        expected = -tau * (
            torch.logsumexp(-values / tau, dim=0)
            - torch.log(torch.tensor(values.numel(), dtype=values.dtype))
        )
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual)


class _ScalarPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        features: torch.Tensor,
        *,
        mask: torch.Tensor,
        asset_indices: torch.Tensor,
        group_ids: torch.Tensor | None,
        costs: torch.Tensor | None,
    ) -> torch.Tensor:
        del asset_indices, group_ids, costs
        return self.scale * features[..., 0] * mask


def _portfolio_training_batch() -> PortfolioSequenceBatch:
    return PortfolioSequenceBatch(
        features=np.array(
            [
                [[[1.0]], [[2.0]], [[-1.0]]],
                [[[-1.0]], [[1.0]], [[2.0]]],
                [[[0.5]], [[-2.0]], [[1.0]]],
                [[[2.0]], [[0.5]], [[-0.5]]],
            ]
        ),
        returns=np.array(
            [
                [[0.01], [0.02], [-0.01]],
                [[-0.02], [0.01], [0.03]],
                [[0.03], [-0.01], [0.02]],
                [[-0.01], [0.03], [0.01]],
            ]
        ),
        vol_scale=np.ones((4, 3, 1)),
    )


def test_portfolio_training_is_invariant_to_forward_chunk_size() -> None:
    batch = _portfolio_training_batch()
    artifacts = []
    for batch_size in (1, batch.batch_size):
        artifacts.append(
            portfolio_runtime.fit_policy_network(
                _ScalarPolicy(),
                batch=batch,
                validation_batch=batch,
                config=PortfolioConfig(
                    batch_size=batch_size,
                    learning_rate=0.01,
                    max_iters=3,
                    eval_every=1,
                    checkpoint_every=1,
                    early_stopping_burn_in_iters=10,
                    use_group_embedding=False,
                    use_cost_in_context=False,
                ),
                device=torch.device("cpu"),
            )
        )

    torch.testing.assert_close(
        artifacts[0].checkpoint_states[3]["scale"],
        artifacts[1].checkpoint_states[3]["scale"],
    )
    assert [entry["train_objective"] for entry in artifacts[0].history] == pytest.approx(
        [entry["train_objective"] for entry in artifacts[1].history]
    )


def test_portfolio_cost_starts_from_supplied_initial_holdings() -> None:
    weights = torch.tensor([[[0.2, -0.1], [0.3, -0.2]]], dtype=torch.float64)
    returns = torch.zeros_like(weights)
    mask = torch.ones_like(weights)
    previous = torch.tensor([[0.1, -0.1]], dtype=torch.float64)

    actual = compute_net_portfolio_returns(
        weights=weights,
        forward_returns=returns,
        vol_scale=torch.ones_like(weights),
        mask=mask,
        costs=torch.ones(2, dtype=torch.float64),
        prev_weights=previous,
        gamma_cost=1.0,
        turnover_penalty=0.5,
    )
    expected = torch.tensor([[-0.075, -0.15]], dtype=torch.float64)

    torch.testing.assert_close(actual, expected)


def test_portfolio_training_rejects_nonfinite_objective() -> None:
    class NonFinitePolicy(_ScalarPolicy):
        def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
            return super().forward(*args, **kwargs) * float("nan")

    batch = _portfolio_training_batch()

    with pytest.raises(FloatingPointError, match="non-finite training loss at step 1"):
        portfolio_runtime.fit_policy_network(
            NonFinitePolicy(),
            batch=batch,
            validation_batch=batch,
            config=PortfolioConfig(max_iters=1, eval_every=1, checkpoint_every=1),
            device=torch.device("cpu"),
        )


def test_portfolio_best_checkpoint_contains_exact_evaluated_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _portfolio_training_batch()
    evaluated_states: list[float] = []

    def fake_validation(
        policy: nn.Module,
        batch: PortfolioSequenceBatch,
        *,
        group_ids: torch.Tensor | None,
        costs: torch.Tensor | None,
        config: PortfolioConfig,
        device: torch.device,
    ) -> float:
        del batch, group_ids, costs, config, device
        evaluated_states.append(float(next(policy.parameters()).detach().item()))
        return float(4 - len(evaluated_states))

    monkeypatch.setattr(portfolio_runtime, "evaluate_pooled_sharpe", fake_validation)
    artifacts = portfolio_runtime.fit_policy_network(
        _ScalarPolicy(),
        batch=batch,
        validation_batch=batch,
        config=PortfolioConfig(
            batch_size=4,
            learning_rate=0.01,
            max_iters=3,
            eval_every=1,
            checkpoint_every=3,
            early_stopping_burn_in_iters=10,
        ),
        device=torch.device("cpu"),
    )

    assert artifacts.best_step == 1
    assert float(artifacts.checkpoint_states[1]["scale"].item()) == pytest.approx(
        evaluated_states[0]
    )


def test_portfolio_constraints_are_satisfied_jointly() -> None:
    raw = np.array([[[10.0, 1.0, -1.0, -2.0]]])
    mask = np.ones_like(raw, dtype=bool)

    constrained = normalize_cross_sectional_weights(
        raw,
        mask=mask,
        gross_exposure=1.0,
        net_exposure=0.2,
        max_abs_weight=0.3,
    )

    assert np.abs(constrained).sum() == pytest.approx(1.0, abs=1e-10)
    assert constrained.sum() == pytest.approx(0.2, abs=1e-10)
    assert np.abs(constrained).max() <= 0.3 + 1e-10


def test_portfolio_constraints_reject_infeasible_request() -> None:
    with pytest.raises(ValueError, match="infeasible portfolio constraints"):
        normalize_cross_sectional_weights(
            np.ones((1, 1, 2)),
            mask=np.ones((1, 1, 2), dtype=bool),
            gross_exposure=1.0,
            net_exposure=0.0,
            max_abs_weight=0.2,
        )
