from __future__ import annotations

import pytest
import torch
from torch import nn

from ml4t.models._internal.sae_nn import SupervisedAutoencoder
from ml4t.models._internal.stochastic_discount_factor_nn import (
    BetaNetwork,
    MomentNetwork,
    StochasticDiscountFactorNetwork,
    compute_sharpe,
)


@pytest.fixture(params=[StochasticDiscountFactorNetwork, MomentNetwork, BetaNetwork])
def context_network(request: pytest.FixtureRequest) -> nn.Module:
    network_type = request.param
    return network_type(n_asset_features=2, n_context_features=1, state_dim=2, dropout=0.0)


def _features() -> torch.Tensor:
    return torch.ones((3, 2, 2))


def test_context_networks_require_context(context_network: nn.Module) -> None:
    with pytest.raises(ValueError, match="context_features must be provided"):
        context_network(_features())


def test_context_networks_require_complete_recurrent_state(context_network: nn.Module) -> None:
    h0 = torch.zeros((1, 1, 2))
    with pytest.raises(ValueError, match="h0 and c0 must be provided together"):
        context_network(_features(), torch.ones((3, 1)), h0=h0)


def test_context_networks_accept_explicit_recurrent_state(context_network: nn.Module) -> None:
    h0 = torch.zeros((1, 1, 2))
    c0 = torch.zeros((1, 1, 2))
    output, state = context_network(_features(), torch.ones((3, 1)), h0=h0, c0=c0)

    assert output.numel() > 0
    assert state[0] is not None
    assert state[1] is not None


@pytest.mark.parametrize(
    "network",
    [
        StochasticDiscountFactorNetwork(n_asset_features=2, hidden_dim=4, dropout=0.0),
        BetaNetwork(n_asset_features=2, hidden_dim=4, dropout=0.0),
    ],
)
def test_cross_section_networks_default_to_all_valid(network: nn.Module) -> None:
    output, state = network(_features())

    assert output.shape == (6,)
    assert state == (None, None)


def test_empty_sdf_portfolio_has_zero_sharpe() -> None:
    assert compute_sharpe(torch.empty(0)).item() == 0.0


def test_supervised_autoencoder_defaults_and_prediction_state() -> None:
    model = SupervisedAutoencoder(n_features=2, n_labels=1)
    model.train()

    betas = model.get_betas(torch.ones((2, 2)))
    prediction = model.predict(torch.ones((2, 2)))

    assert betas.shape == (2, 96)
    assert prediction.shape == (2, 1)
    assert model.training


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_units": (1,)}, "hidden_units must contain 6 entries"),
        ({"dropout_rates": (0.0,)}, "dropout_rates must contain 8 entries"),
        ({"output_activation": "softmax"}, "output_activation must be one of"),
    ],
)
def test_supervised_autoencoder_rejects_invalid_architecture(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SupervisedAutoencoder(n_features=2, **kwargs)
