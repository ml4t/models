"""Config dataclasses for end-to-end portfolio learners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ml4t.models.configs.base import (
    BaseModelConfig,
    _require_int,
    _require_probability,
    _require_real,
)

__all__ = ["DeepPortfolioConfig", "LinearPortfolioConfig", "LSTMPortfolioConfig", "PortfolioConfig"]


@dataclass(frozen=True, slots=True)
class PortfolioConfig(BaseModelConfig):
    """Base config for portfolio-learning models."""

    model_name: ClassVar[str] = "portfolio_model"
    turnover_penalty: float = 0.0
    dropout: float = 0.1

    asset_embedding_dim: int = 8
    group_embedding_dim: int = 4
    use_group_embedding: bool = False
    use_cost_in_context: bool = False
    vvsn_hidden_dim: int = 64

    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    annualization_factor: float = 252.0
    sharpe_eps: float = 1e-8
    gamma_cost: float = 0.5
    softmin_tau: float = 0.2
    softmin_lambda: float = 0.1
    burn_in: int = 0
    max_iters: int = 200
    eval_every: int = 10
    metric_ema_alpha: float = 0.45
    metric_min_delta: float = 0.001
    early_stopping_patience: int = 20
    early_stopping_burn_in_iters: int = 20
    checkpoint_every: int = 10
    checkpoint_steps: tuple[int, ...] = ()
    default_checkpoint: int | None = None

    def __post_init__(self) -> None:
        BaseModelConfig.__post_init__(self)
        _require_real("turnover_penalty", self.turnover_penalty, minimum=0.0)
        _require_probability("dropout", self.dropout)
        for name in ("asset_embedding_dim", "group_embedding_dim", "vvsn_hidden_dim", "batch_size"):
            _require_int(name, getattr(self, name))
        _require_real("learning_rate", self.learning_rate, minimum=0.0, inclusive=False)
        _require_real("weight_decay", self.weight_decay, minimum=0.0)
        _require_real("max_grad_norm", self.max_grad_norm, minimum=0.0, inclusive=False)
        _require_real(
            "annualization_factor", self.annualization_factor, minimum=0.0, inclusive=False
        )
        _require_real("sharpe_eps", self.sharpe_eps, minimum=0.0, inclusive=False)
        _require_real("gamma_cost", self.gamma_cost, minimum=0.0)
        _require_real("softmin_tau", self.softmin_tau, minimum=0.0, inclusive=False)
        _require_real("softmin_lambda", self.softmin_lambda, minimum=0.0)
        _require_int("burn_in", self.burn_in, minimum=0)
        _require_int("max_iters", self.max_iters)
        _require_int("eval_every", self.eval_every)
        if not 0.0 < self.metric_ema_alpha <= 1.0:
            raise ValueError(f"metric_ema_alpha must be in (0, 1]; got {self.metric_ema_alpha!r}")
        _require_real("metric_min_delta", self.metric_min_delta, minimum=0.0)
        _require_int("early_stopping_patience", self.early_stopping_patience)
        _require_int("early_stopping_burn_in_iters", self.early_stopping_burn_in_iters, minimum=0)
        _require_int("checkpoint_every", self.checkpoint_every)
        if len(set(self.checkpoint_steps)) != len(self.checkpoint_steps):
            raise ValueError("checkpoint_steps must not contain duplicates")
        for checkpoint in self.checkpoint_steps:
            _require_int("checkpoint_steps entry", checkpoint)
            if checkpoint > self.max_iters:
                raise ValueError(
                    f"checkpoint_steps entries must be <= {self.max_iters}; got {checkpoint}"
                )
        if self.default_checkpoint is not None:
            _require_int("default_checkpoint", self.default_checkpoint)
            if self.default_checkpoint > self.max_iters:
                raise ValueError(
                    f"default_checkpoint must be <= {self.max_iters}; got {self.default_checkpoint}"
                )


@dataclass(frozen=True, slots=True)
class LSTMPortfolioConfig(PortfolioConfig):
    """Starter config for a sequence-based portfolio learner."""

    model_name: ClassVar[str] = "lstm_portfolio"
    hidden_size: int = 64
    n_layers: int = 1

    def __post_init__(self) -> None:
        PortfolioConfig.__post_init__(self)
        _require_int("hidden_size", self.hidden_size)
        _require_int("n_layers", self.n_layers)


@dataclass(frozen=True, slots=True)
class LinearPortfolioConfig(PortfolioConfig):
    """Config for a pooled linear feature portfolio baseline."""

    model_name: ClassVar[str] = "linear_portfolio"
    ridge_alpha: float = 1e-4
    fit_intercept: bool = True
    gross_exposure: float = 1.0
    net_exposure: float = 0.0
    max_abs_weight: float | None = None

    def __post_init__(self) -> None:
        PortfolioConfig.__post_init__(self)
        _require_real("ridge_alpha", self.ridge_alpha, minimum=0.0)
        _require_real("gross_exposure", self.gross_exposure, minimum=0.0, inclusive=False)
        if not -self.gross_exposure <= self.net_exposure <= self.gross_exposure:
            raise ValueError(
                "net_exposure must be within [-gross_exposure, gross_exposure]; "
                f"got net_exposure={self.net_exposure}, gross_exposure={self.gross_exposure}"
            )
        if self.max_abs_weight is not None:
            _require_real("max_abs_weight", self.max_abs_weight, minimum=0.0, inclusive=False)


@dataclass(frozen=True, slots=True)
class DeepPortfolioConfig(PortfolioConfig):
    """Config for DeePM-style end-to-end portfolio learners."""

    model_name: ClassVar[str] = "deep_portfolio"

    d_model: int = 64
    n_heads: int = 2

    lstm_layers: int = 1
    temporal_mha_layers: int = 1
    cross_attention_heads: int = 2
    cross_attention_lag: int = 1
    macro_gnn_heads: int = 2

    adapter_hidden_mult: int = 2

    def __post_init__(self) -> None:
        PortfolioConfig.__post_init__(self)
        for name in (
            "d_model",
            "n_heads",
            "lstm_layers",
            "temporal_mha_layers",
            "cross_attention_heads",
            "macro_gnn_heads",
            "adapter_hidden_mult",
        ):
            _require_int(name, getattr(self, name))
        _require_int("cross_attention_lag", self.cross_attention_lag, minimum=0)
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads; got {self.d_model} and {self.n_heads}"
            )
        if self.d_model % self.cross_attention_heads != 0:
            raise ValueError(
                "d_model must be divisible by cross_attention_heads; "
                f"got {self.d_model} and {self.cross_attention_heads}"
            )
        if self.d_model % self.macro_gnn_heads != 0:
            raise ValueError(
                "d_model must be divisible by macro_gnn_heads; "
                f"got {self.d_model} and {self.macro_gnn_heads}"
            )
