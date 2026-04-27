"""Neural components for stochastic discount factor models."""

from __future__ import annotations

import torch
import torch.nn as nn


class SDFNetwork(nn.Module):
    """Learn cross-sectional SDF weights from characteristics and optional context."""

    def __init__(
        self,
        n_asset_features: int,
        n_context_features: int = 0,
        state_dim: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.use_context = n_context_features > 0

        if self.use_context:
            self.lstm = nn.LSTM(
                input_size=n_context_features,
                hidden_size=state_dim,
                batch_first=True,
            )

        ffn_input_dim = n_asset_features + (state_dim if self.use_context else 0)
        self.ffn = nn.Sequential(
            nn.Linear(ffn_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        asset_features: torch.Tensor,
        context_features: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        h0: torch.Tensor | None = None,
        c0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, torch.Tensor | None]]:
        if self.use_context and context_features is None:
            raise ValueError("context_features must be provided when n_context_features > 0")
        if (h0 is None) != (c0 is None):
            raise ValueError("h0 and c0 must be provided together")

        n_periods, n_assets, _ = asset_features.shape
        if mask is None:
            mask = torch.ones(n_periods, n_assets, dtype=torch.bool, device=asset_features.device)

        if self.use_context and context_features is not None:
            if h0 is None:
                h0 = torch.zeros(1, 1, self.state_dim, device=asset_features.device)
                c0 = torch.zeros(1, 1, self.state_dim, device=asset_features.device)
            context_seq = context_features.unsqueeze(0)
            context_states, (h_n, c_n) = self.lstm(context_seq, (h0, c0))
            context_states = context_states.squeeze(0)
            context_tiled = context_states.unsqueeze(1).expand(-1, n_assets, -1)
            ffn_input = torch.cat([asset_features[mask], context_tiled[mask]], dim=1)
        else:
            ffn_input = asset_features[mask]
            h_n = c_n = None

        weights = self.ffn(ffn_input).squeeze(-1)
        return weights, (h_n, c_n)


class MomentNetwork(nn.Module):
    """Adversarial moment network that learns instruments."""

    def __init__(
        self,
        n_asset_features: int,
        n_context_features: int = 0,
        n_instruments: int = 8,
        state_dim: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_instruments = n_instruments
        self.use_context = n_context_features > 0

        if self.use_context:
            self.lstm = nn.LSTM(
                input_size=n_context_features,
                hidden_size=state_dim,
                batch_first=True,
            )

        ffn_input_dim = n_asset_features + (state_dim if self.use_context else 0)
        self.ffn = nn.Sequential(
            nn.Linear(ffn_input_dim, n_instruments),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        asset_features: torch.Tensor,
        context_features: torch.Tensor | None = None,
        h0: torch.Tensor | None = None,
        c0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, torch.Tensor | None]]:
        if self.use_context and context_features is None:
            raise ValueError("context_features must be provided when n_context_features > 0")
        if (h0 is None) != (c0 is None):
            raise ValueError("h0 and c0 must be provided together")

        n_periods, n_assets, _ = asset_features.shape
        if self.use_context and context_features is not None:
            if h0 is None:
                h0 = torch.zeros(1, 1, self.state_dim, device=asset_features.device)
                c0 = torch.zeros(1, 1, self.state_dim, device=asset_features.device)
            context_seq = context_features.unsqueeze(0)
            context_states, (h_n, c_n) = self.lstm(context_seq, (h0, c0))
            context_states = context_states.squeeze(0)
            context_tiled = context_states.unsqueeze(1).expand(-1, n_assets, -1)
            ffn_input = torch.cat([asset_features, context_tiled], dim=2)
        else:
            ffn_input = asset_features
            h_n = c_n = None

        instruments = self.ffn(ffn_input)
        return instruments.permute(2, 0, 1), (h_n, c_n)


def get_segment_ids(mask: torch.Tensor) -> torch.Tensor:
    """Map each valid observation to its time index."""

    n_periods, n_assets = mask.shape
    time_ids = torch.arange(n_periods, device=mask.device).unsqueeze(1).expand(-1, n_assets)
    return time_ids[mask]


def construct_sdf(
    returns: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Construct the dated SDF series from cross-sectional weights."""

    weighted_returns = weights * returns[mask]
    sdf_values = torch.zeros(returns.shape[0], device=weights.device, dtype=weights.dtype)
    sdf_values.scatter_add_(0, get_segment_ids(mask), weighted_returns)
    return 1.0 + sdf_values


def unconditional_loss(
    weights: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    n_obs_per_asset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unconditional pricing loss with a constant instrument."""

    sdf = construct_sdf(returns, weights, mask)
    sample_moments = (returns * mask.float() * sdf.unsqueeze(1)).unsqueeze(0)
    weighted_moments = sample_moments.sum(dim=1) / n_obs_per_asset.clamp(min=1).unsqueeze(0)
    n_obs_norm = n_obs_per_asset / n_obs_per_asset.max().clamp(min=1)
    loss = (weighted_moments.pow(2) * n_obs_norm).mean()
    return loss, sdf


def conditional_loss(
    weights: torch.Tensor,
    instruments: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    n_obs_per_asset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conditional pricing loss with learned instruments."""

    n_instruments = instruments.shape[0]
    sdf = construct_sdf(returns, weights, mask)
    sample_moments = returns * mask.float() * sdf.unsqueeze(1) * instruments
    weighted_moments = sample_moments.sum(dim=1) / n_obs_per_asset.clamp(min=1).unsqueeze(0)
    n_obs_norm = n_obs_per_asset / n_obs_per_asset.max().clamp(min=1)
    tiled = n_obs_norm.unsqueeze(0).expand(n_instruments, -1)
    loss = (weighted_moments.pow(2) * tiled).mean()
    return loss, sdf


def compute_sharpe(sdf: torch.Tensor) -> torch.Tensor:
    """Sharpe ratio of the portfolio induced by the SDF."""

    portfolio_return = 1.0 - sdf
    if portfolio_return.numel() == 0:
        return torch.zeros((), device=sdf.device, dtype=sdf.dtype)
    mean = portfolio_return.mean()
    std = portfolio_return.std(unbiased=False).clamp(min=1e-8)
    out = mean / std
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))
