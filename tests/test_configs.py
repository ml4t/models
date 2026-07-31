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
        (PCAConfig, {"seed": True}, "seed"),
        (PCAConfig, {"n_factors": 0}, "n_factors"),
        (RPPCAConfig, {"gamma": float("nan")}, "gamma"),
        (RPPCAConfig, {"base_moment": "invalid"}, "base_moment"),
        (RPPCAConfig, {"normalize_loadings": "invalid"}, "normalize_loadings"),
        (IPCAConfig, {"max_iter": 0}, "max_iter"),
        (IPCAConfig, {"tol": 0.0}, "tol"),
        (CAEConfig, {"task_type": "invalid"}, "task_type"),
        (CAEConfig, {"hidden_units": (0,)}, "hidden_units"),
        (CAEConfig, {"n_epochs": 0}, "n_epochs"),
        (CAEConfig, {"checkpoint_epochs": (1, 1)}, "checkpoint_epochs"),
        (CAEConfig, {"checkpoint_epochs": (1.5,)}, "checkpoint_epochs entry"),
        (CAEConfig, {"n_epochs": 2, "checkpoint_epochs": (3,)}, "checkpoint_epochs entries"),
        (CAEConfig, {"default_checkpoint": 1.5}, "default_checkpoint"),
        (CAEConfig, {"n_epochs": 2, "default_checkpoint": 3}, "default_checkpoint"),
        (SAEConfig, {"task_type": "invalid"}, "task_type"),
        (SAEConfig, {"main_hidden_units": (1, 2, 3)}, "main_hidden_units"),
        (SAEConfig, {"lr": -1.0}, "lr"),
        (SAEConfig, {"dropout_rates": (0.0, 1.0)}, "dropout_rates"),
        (EWMABaseForecasterConfig, {"half_life": -5.0}, "half_life"),
        (LinearPortfolioConfig, {"ridge_alpha": -1.0}, "ridge_alpha"),
        (LinearPortfolioConfig, {"gross_exposure": 0.5, "net_exposure": 1.0}, "net_exposure"),
        (LinearPortfolioConfig, {"max_abs_weight": 0.0}, "max_abs_weight"),
        (LinearPortfolioConfig, {"metric_ema_alpha": 0.0}, "metric_ema_alpha"),
        (LinearPortfolioConfig, {"checkpoint_steps": (1, 1)}, "checkpoint_steps"),
        (LinearPortfolioConfig, {"checkpoint_steps": (1.5,)}, "checkpoint_steps entry"),
        (
            LinearPortfolioConfig,
            {"max_iters": 2, "checkpoint_steps": (3,)},
            "checkpoint_steps entries",
        ),
        (LinearPortfolioConfig, {"default_checkpoint": 1.5}, "default_checkpoint"),
        (
            LinearPortfolioConfig,
            {"max_iters": 2, "default_checkpoint": 3},
            "default_checkpoint",
        ),
        (LSTMPortfolioConfig, {"max_iters": 0}, "max_iters"),
        (DeepPortfolioConfig, {"d_model": 7, "n_heads": 2}, "d_model"),
        (DeepPortfolioConfig, {"d_model": 6, "cross_attention_heads": 4}, "d_model"),
        (DeepPortfolioConfig, {"d_model": 6, "macro_gnn_heads": 4}, "d_model"),
        (StochasticDiscountFactorConfig, {"dropout": 1.0}, "dropout"),
        (StochasticDiscountFactorConfig, {"device": "not-a-device"}, "device"),
        (StochasticDiscountFactorConfig, {"dtype": "float16"}, "dtype"),
        (
            StochasticDiscountFactorConfig,
            {"default_checkpoint": ("invalid", 1)},
            "phase is invalid",
        ),
        (
            StochasticDiscountFactorConfig,
            {"default_checkpoint": ("conditional", 1.5)},
            "epoch",
        ),
        (
            StochasticDiscountFactorConfig,
            {"n_epochs_cond": 2, "default_checkpoint": ("conditional", 3)},
            "epoch must be <= 2",
        ),
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
