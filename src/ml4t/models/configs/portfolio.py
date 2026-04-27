"""Config dataclasses for end-to-end portfolio learners."""

from __future__ import annotations

from dataclasses import dataclass

from ml4t.models.configs.base import BaseModelConfig


@dataclass(frozen=True, slots=True)
class PortfolioConfig(BaseModelConfig):
    """Base config for portfolio-learning models."""

    model_name: str = "portfolio_model"
    turnover_penalty: float = 0.0
    dropout: float = 0.1

    asset_embedding_dim: int = 8
    group_embedding_dim: int = 4
    use_group_embedding: bool = True
    use_cost_in_context: bool = True
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


@dataclass(frozen=True, slots=True)
class LSTMPortfolioConfig(PortfolioConfig):
    """Starter config for a sequence-based portfolio learner."""

    model_name: str = "lstm_portfolio"
    hidden_size: int = 64
    n_layers: int = 1


@dataclass(frozen=True, slots=True)
class LinearPortfolioConfig(PortfolioConfig):
    """Config for a pooled linear feature portfolio baseline."""

    model_name: str = "linear_portfolio"
    ridge_alpha: float = 1e-4
    fit_intercept: bool = True
    gross_exposure: float = 1.0
    net_exposure: float = 0.0
    max_abs_weight: float | None = None


@dataclass(frozen=True, slots=True)
class DeepPortfolioConfig(PortfolioConfig):
    """Config for DeePM-style end-to-end portfolio learners."""

    model_name: str = "deep_portfolio"

    d_model: int = 64
    n_heads: int = 2

    lstm_layers: int = 1
    temporal_mha_layers: int = 1
    cross_attention_heads: int = 2
    cross_attention_lag: int = 1
    macro_gnn_heads: int = 2

    adapter_hidden_mult: int = 2
