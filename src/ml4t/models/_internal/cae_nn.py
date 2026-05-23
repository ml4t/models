"""Conditional-autoencoder network components."""

from __future__ import annotations

import torch
import torch.nn as nn


class BetaNetwork(nn.Module):
    """Map characteristics to per-asset factor loadings."""

    def __init__(
        self, n_characteristics: int, n_factors: int, hidden_units: tuple[int, ...]
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = n_characteristics
        for units in hidden_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            in_features = units
        layers.append(nn.Linear(in_features, n_factors))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FactorNetwork(nn.Module):
    """Linear factor network from managed portfolios to factor returns."""

    def __init__(self, n_instruments: int, n_factors: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_instruments, n_factors, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ConditionalAutoencoder(nn.Module):
    """Conditional autoencoder with neural betas and linear factors."""

    def __init__(
        self,
        n_characteristics: int,
        n_instruments: int,
        n_factors: int,
        hidden_units: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.beta_net = BetaNetwork(n_characteristics, n_factors, hidden_units)
        self.factor_net = FactorNetwork(n_instruments, n_factors)

    def forward(self, characteristics: torch.Tensor, portfolios: torch.Tensor) -> torch.Tensor:
        betas = self.beta_net(characteristics)
        factors = self.factor_net(portfolios)
        return (betas * factors).sum(dim=1)

    def get_betas(self, characteristics: torch.Tensor) -> torch.Tensor:
        return self.beta_net(characteristics)

    def get_factors(self, portfolios: torch.Tensor) -> torch.Tensor:
        return self.factor_net(portfolios)


def l1_regularization(model: ConditionalAutoencoder, lambda_l1: float) -> torch.Tensor:
    """Apply L1 regularization to hidden beta-network weights."""
    layers = [module for module in model.beta_net.network if isinstance(module, nn.Linear)]
    if len(layers) <= 1:
        return torch.zeros((), device=next(model.parameters()).device)
    hidden_layers = layers[:-1]
    l1 = torch.zeros((), device=next(model.parameters()).device)
    for layer in hidden_layers:
        l1 = l1 + layer.weight.abs().sum()
    return lambda_l1 * l1
