"""Config dataclasses for latent-factor estimators."""

from __future__ import annotations

from dataclasses import dataclass

from ml4t.models.configs.base import BaseModelConfig


@dataclass(frozen=True, slots=True)
class LatentFactorConfig(BaseModelConfig):
    """Shared latent-factor configuration."""

    model_name: str = "latent_factor"
    n_factors: int = 5
    persistent_entities: bool = False


@dataclass(frozen=True, slots=True)
class PCAConfig(LatentFactorConfig):
    """Config for PCA and related persistent-panel baselines."""

    model_name: str = "pca"
    persistent_entities: bool = True


@dataclass(frozen=True, slots=True)
class RPPCAConfig(LatentFactorConfig):
    """Config for risk-premium-aware PCA."""

    model_name: str = "rp_pca"
    persistent_entities: bool = True
    gamma: float = 0.0
    base_moment: str = "covariance"
    scale_by_asset_volatility: bool = False
    normalize_loadings: str = "unit_length"
    orthogonalize_factors: bool = False


@dataclass(frozen=True, slots=True)
class IPCAConfig(LatentFactorConfig):
    """Config for IPCA."""

    model_name: str = "ipca"
    max_iter: int = 100
    tol: float = 1e-6
    factor_ridge: float = 1e-6
    gamma_ridge: float = 1e-6


@dataclass(frozen=True, slots=True)
class CAEConfig(LatentFactorConfig):
    """Config for conditional autoencoders."""

    model_name: str = "cae"
    task_type: str = "regression"
    hidden_units: tuple[int, ...] = (32,)
    n_ensemble: int = 1
    n_epochs: int = 50
    checkpoint_interval: int | None = 5
    checkpoint_epochs: tuple[int, ...] = ()
    default_checkpoint: int | None = None
    lr: float = 1e-3
    lambda_l1: float = 1e-4


@dataclass(frozen=True, slots=True)
class StochasticDiscountFactorConfig(BaseModelConfig):
    """Config for stochastic discount factor networks."""

    model_name: str = "stochastic_discount_factor"
    output_mode: str = "weights"
    state_dim_sdf: int = 4
    state_dim_moment: int = 32
    hidden_dim: int = 64
    n_instruments: int = 8
    dropout: float = 0.05
    n_epochs_unc: int = 256
    n_epochs_moment: int = 64
    n_epochs_cond: int = 1024
    checkpoint_interval: int | None = 5
    checkpoint_epochs: tuple[int, ...] = ()
    default_checkpoint: int | None = None
    expected_return_mapper: str = "linear"
    beta_state_dim: int = 4
    beta_hidden_dim: int = 64
    beta_n_epochs: int = 256
    beta_checkpoint_interval: int | None = 16
    beta_checkpoint_epochs: tuple[int, ...] = ()
    beta_default_checkpoint: int | None = None
    beta_lr: float = 1e-3
    burn_in_epochs: int = 0
    lr: float = 1e-3
    weight_decay: float = 0.0
