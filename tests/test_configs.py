from __future__ import annotations

import pytest

from ml4t.models import (
    CAEConfig,
    DeepPortfolioConfig,
    EWMABaseForecasterConfig,
    IPCAConfig,
    LinearPortfolioConfig,
    LSTMPortfolioConfig,
    PCAConfig,
    RPPCAConfig,
    SAEConfig,
    StochasticDiscountFactorConfig,
)


@pytest.mark.parametrize(
    ("config_type", "kwargs", "match"),
    [
        (PCAConfig, {"n_factors": 0}, "n_factors"),
        (RPPCAConfig, {"gamma": float("nan")}, "gamma"),
        (RPPCAConfig, {"base_moment": "invalid"}, "base_moment"),
        (IPCAConfig, {"max_iter": 0}, "max_iter"),
        (IPCAConfig, {"tol": 0.0}, "tol"),
        (CAEConfig, {"task_type": "invalid"}, "task_type"),
        (CAEConfig, {"n_epochs": 0}, "n_epochs"),
        (SAEConfig, {"lr": -1.0}, "lr"),
        (SAEConfig, {"dropout_rates": (0.0, 1.0)}, "dropout_rates"),
        (EWMABaseForecasterConfig, {"half_life": -5.0}, "half_life"),
        (LinearPortfolioConfig, {"ridge_alpha": -1.0}, "ridge_alpha"),
        (LinearPortfolioConfig, {"gross_exposure": 0.5, "net_exposure": 1.0}, "net_exposure"),
        (LSTMPortfolioConfig, {"max_iters": 0}, "max_iters"),
        (DeepPortfolioConfig, {"d_model": 7, "n_heads": 2}, "d_model"),
        (StochasticDiscountFactorConfig, {"dropout": 1.0}, "dropout"),
        (StochasticDiscountFactorConfig, {"device": "not-a-device"}, "device"),
        (StochasticDiscountFactorConfig, {"dtype": "float16"}, "dtype"),
    ],
)
def test_public_configs_reject_invalid_values(
    config_type: type[object],
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        config_type(**kwargs)


@pytest.mark.parametrize(
    "config",
    [
        PCAConfig(),
        RPPCAConfig(),
        IPCAConfig(),
        CAEConfig(),
        SAEConfig(),
        EWMABaseForecasterConfig(),
        LinearPortfolioConfig(),
        LSTMPortfolioConfig(),
        DeepPortfolioConfig(),
        StochasticDiscountFactorConfig(),
    ],
)
def test_model_identity_is_not_constructor_overridable(config: object) -> None:
    config_type = type(config)
    with pytest.raises(TypeError, match="model_name"):
        config_type(model_name="different")


def test_neural_portfolio_defaults_do_not_require_optional_context() -> None:
    assert not LSTMPortfolioConfig().use_group_embedding
    assert not LSTMPortfolioConfig().use_cost_in_context
    assert not DeepPortfolioConfig().use_group_embedding
    assert not DeepPortfolioConfig().use_cost_in_context
