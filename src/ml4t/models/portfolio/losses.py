"""Losses for end-to-end portfolio learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PortfolioLossOutput:
    """Outputs returned by the robust Sharpe objective."""

    loss: torch.Tensor
    sharpe_pool: torch.Tensor
    softmin_sharpe: torch.Tensor
    objective: torch.Tensor
    net_returns: torch.Tensor


def compute_net_portfolio_returns(
    *,
    weights: torch.Tensor,
    forward_returns: torch.Tensor,
    vol_scale: torch.Tensor,
    mask: torch.Tensor,
    costs: torch.Tensor | None,
    gamma_cost: float,
) -> torch.Tensor:
    """Compute net portfolio returns for a sequence of cross-sectional decisions."""

    batch_size, n_periods, n_assets = weights.shape
    available = mask.to(dtype=torch.bool)

    scaled_weights = torch.where(
        available,
        vol_scale * weights,
        torch.zeros_like(weights),
    )
    previous = torch.cat(
        [
            torch.zeros((batch_size, 1, n_assets), device=weights.device, dtype=weights.dtype),
            scaled_weights[:, :-1, :],
        ],
        dim=1,
    )

    gross_contributions = torch.where(
        available,
        weights * forward_returns,
        torch.zeros_like(weights),
    )
    gross = gross_contributions.sum(dim=-1)
    n_available = mask.sum(dim=-1).clamp(min=1.0)
    gross = gross / n_available

    if costs is None:
        return gross

    if costs.ndim == 2 and costs.shape[1] == 1 or costs.ndim == 1 and costs.shape[0] == n_assets:
        cost_tensor = costs.view(1, 1, n_assets)
    else:
        raise ValueError("costs must have shape (N,) or (N, 1)")

    turnover = torch.where(
        available,
        torch.abs(scaled_weights - previous),
        torch.zeros_like(weights),
    )
    cost = (cost_tensor * turnover).sum(dim=-1)
    cost = (gamma_cost * cost) / n_available
    return gross - cost


def sharpe_ratio(
    returns: torch.Tensor,
    *,
    annualization_factor: float,
    eps: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Compute a differentiable Sharpe ratio."""

    if dim is None:
        mu = returns.mean()
        var = returns.var(unbiased=False)
    else:
        mu = returns.mean(dim=dim)
        var = returns.var(dim=dim, unbiased=False)
    return (annualization_factor**0.5) * mu / torch.sqrt(var + eps)


def softmin_sharpe(window_sharpes: torch.Tensor, *, tau: float) -> torch.Tensor:
    """Soft minimum of window-wise Sharpe ratios."""

    n_windows = torch.as_tensor(
        window_sharpes.numel(),
        dtype=window_sharpes.dtype,
        device=window_sharpes.device,
    )
    return -tau * (torch.logsumexp(-window_sharpes / tau, dim=0) - torch.log(n_windows))


def robust_sharpe_loss(
    *,
    weights: torch.Tensor,
    forward_returns: torch.Tensor,
    vol_scale: torch.Tensor,
    mask: torch.Tensor,
    costs: torch.Tensor | None,
    burn_in: int,
    gamma_cost: float,
    annualization_factor: float,
    eps: float,
    tau: float,
    lambda_soft: float,
) -> PortfolioLossOutput:
    """Compute a pooled-plus-softmin Sharpe objective."""

    net_returns = compute_net_portfolio_returns(
        weights=weights,
        forward_returns=forward_returns,
        vol_scale=vol_scale,
        mask=mask,
        costs=costs,
        gamma_cost=gamma_cost,
    )
    effective_returns = net_returns[:, burn_in:] if burn_in > 0 else net_returns

    pooled_sharpe = sharpe_ratio(
        effective_returns.reshape(-1),
        annualization_factor=annualization_factor,
        eps=eps,
    )
    window_sharpes = sharpe_ratio(
        effective_returns,
        annualization_factor=annualization_factor,
        eps=eps,
        dim=1,
    )
    softmin = softmin_sharpe(window_sharpes, tau=tau)
    objective = pooled_sharpe + lambda_soft * softmin
    return PortfolioLossOutput(
        loss=-objective,
        sharpe_pool=pooled_sharpe.detach(),
        softmin_sharpe=softmin.detach(),
        objective=objective.detach(),
        net_returns=net_returns.detach(),
    )
