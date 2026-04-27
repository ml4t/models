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
class SAEConfig(LatentFactorConfig):
    """Config for supervised autoencoders."""

    model_name: str = "sae"
    task_type: str = "regression"
    hidden_units: tuple[int, ...] | None = None
    dropout_rates: tuple[float, ...] | None = None
    noise_std: float = 0.035
    alpha: float = 1.0
    aux_weight: float = 1.0
    n_ensemble: int = 1
    n_epochs: int = 50
    checkpoint_interval: int | None = 5
    checkpoint_epochs: tuple[int, ...] = ()
    default_checkpoint: int | None = None
    lr: float = 1e-4
    checkpoint_interval: int | None = 5
    checkpoint_epochs: tuple[int, ...] = ()
    default_checkpoint: int | None = None


@dataclass(frozen=True, slots=True)
class SDFConfig(LatentFactorConfig):
    """Config for SDF networks."""

    model_name: str = "sdf"
    output_mode: str = "weights"
