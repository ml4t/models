"""Config dataclasses for latent-factor estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ml4t.models.configs.base import (
    BaseModelConfig,
    _require_finite,
    _require_int,
    _require_probability,
    _require_real,
    _validate_checkpoint_schedule,
)

__all__ = [
    "CAEConfig",
    "IPCAConfig",
    "LatentFactorConfig",
    "PCAConfig",
    "RPPCAConfig",
    "StochasticDiscountFactorConfig",
]


@dataclass(frozen=True, slots=True)
class LatentFactorConfig(BaseModelConfig):
    """Shared latent-factor configuration."""

    model_name: ClassVar[str] = "latent_factor"
    n_factors: int = 5
    persistent_entities: ClassVar[bool] = False

    def __post_init__(self) -> None:
        BaseModelConfig.__post_init__(self)
        _require_int("n_factors", self.n_factors)


@dataclass(frozen=True, slots=True)
class PCAConfig(LatentFactorConfig):
    """Config for PCA and related persistent-panel baselines."""

    model_name: ClassVar[str] = "pca"
    persistent_entities: ClassVar[bool] = True


@dataclass(frozen=True, slots=True)
class RPPCAConfig(LatentFactorConfig):
    """Config for risk-premium-aware PCA."""

    model_name: ClassVar[str] = "rp_pca"
    persistent_entities: ClassVar[bool] = True
    gamma: float = 0.0
    base_moment: str = "second_moment"
    scale_by_asset_volatility: bool = False
    normalize_loadings: str = "unit_length"
    orthogonalize_factors: bool = False

    def __post_init__(self) -> None:
        LatentFactorConfig.__post_init__(self)
        _require_finite("gamma", self.gamma)
        if self.base_moment not in {"covariance", "second_moment"}:
            raise ValueError(
                f"base_moment must be 'covariance' or 'second_moment'; got {self.base_moment!r}"
            )
        if self.normalize_loadings not in {"unit_length", "variance"}:
            raise ValueError(
                "normalize_loadings must be 'unit_length' or 'variance'; "
                f"got {self.normalize_loadings!r}"
            )


@dataclass(frozen=True, slots=True)
class IPCAConfig(LatentFactorConfig):
    """Config for IPCA."""

    model_name: ClassVar[str] = "ipca"
    max_iter: int = 10_000
    tol: float = 1e-6
    factor_ridge: float = 1e-6
    gamma_ridge: float = 1e-6

    def __post_init__(self) -> None:
        LatentFactorConfig.__post_init__(self)
        _require_int("max_iter", self.max_iter)
        _require_real("tol", self.tol, minimum=0.0, inclusive=False)
        _require_real("factor_ridge", self.factor_ridge, minimum=0.0)
        _require_real("gamma_ridge", self.gamma_ridge, minimum=0.0)


@dataclass(frozen=True, slots=True)
class CAEConfig(LatentFactorConfig):
    """Config for conditional autoencoders."""

    model_name: ClassVar[str] = "cae"
    task_type: str = "regression"
    hidden_units: tuple[int, ...] = (32,)
    n_ensemble: int = 1
    n_epochs: int = 50
    checkpoint_interval: int | None = 5
    checkpoint_epochs: tuple[int, ...] = ()
    default_checkpoint: int | None = None
    lr: float = 1e-3
    lambda_l1: float = 1e-4
    batch_size: int = 10_000

    def __post_init__(self) -> None:
        LatentFactorConfig.__post_init__(self)
        if self.task_type not in {"regression", "classification"}:
            raise ValueError(
                f"task_type must be 'regression' or 'classification'; got {self.task_type!r}"
            )
        if any(type(unit) is not int or unit < 1 for unit in self.hidden_units):
            raise ValueError(
                f"hidden_units must contain positive integers; got {self.hidden_units!r}"
            )
        _require_int("n_ensemble", self.n_ensemble)
        _require_int("n_epochs", self.n_epochs)
        _require_int("batch_size", self.batch_size)
        _require_real("lr", self.lr, minimum=0.0, inclusive=False)
        _require_real("lambda_l1", self.lambda_l1, minimum=0.0)
        _validate_checkpoint_schedule(
            total=self.n_epochs,
            interval=self.checkpoint_interval,
            checkpoints=self.checkpoint_epochs,
            default=self.default_checkpoint,
        )


@dataclass(frozen=True, slots=True)
class StochasticDiscountFactorConfig(BaseModelConfig):
    """Config for stochastic discount factor networks."""

    model_name: ClassVar[str] = "stochastic_discount_factor"
    output_mode: ClassVar[str] = "weights"
    state_dim_sdf: int = 4
    state_dim_moment: int = 32
    hidden_dim: int = 64
    n_instruments: int = 8
    dropout: float = 0.05
    n_epochs_unc: int = 256
    n_epochs_moment: int = 64
    n_epochs_cond: int = 1024
    checkpoint_interval: int | None = None
    checkpoint_epochs: tuple[int, ...] = ()
    default_checkpoint: int | tuple[str, int] | None = None
    expected_return_mapper: ClassVar[str] = "linear"
    beta_state_dim: int = 4
    beta_hidden_dim: int = 64
    beta_n_epochs: int = 256
    beta_checkpoint_interval: int | None = None
    beta_checkpoint_epochs: tuple[int, ...] = ()
    beta_default_checkpoint: int | None = None
    beta_lr: float = 1e-3
    burn_in_epochs: int = 0
    lr: float = 1e-3
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        BaseModelConfig.__post_init__(self)
        for name in (
            "state_dim_sdf",
            "state_dim_moment",
            "hidden_dim",
            "n_instruments",
            "n_epochs_unc",
            "n_epochs_moment",
            "n_epochs_cond",
            "beta_state_dim",
            "beta_hidden_dim",
            "beta_n_epochs",
        ):
            _require_int(name, getattr(self, name))
        _require_probability("dropout", self.dropout)
        _require_int("burn_in_epochs", self.burn_in_epochs, minimum=0)
        _require_real("lr", self.lr, minimum=0.0, inclusive=False)
        _require_real("beta_lr", self.beta_lr, minimum=0.0, inclusive=False)
        _require_real("weight_decay", self.weight_decay, minimum=0.0)
        conditional_default = (
            self.default_checkpoint if isinstance(self.default_checkpoint, int) else None
        )
        _validate_checkpoint_schedule(
            total=self.n_epochs_cond,
            interval=self.checkpoint_interval,
            checkpoints=self.checkpoint_epochs,
            default=conditional_default,
        )
        _validate_checkpoint_schedule(
            total=self.beta_n_epochs,
            interval=self.beta_checkpoint_interval,
            checkpoints=self.beta_checkpoint_epochs,
            default=self.beta_default_checkpoint,
            prefix="beta_",
        )
        if isinstance(self.default_checkpoint, tuple):
            phase, checkpoint = self.default_checkpoint
            phase_totals = {
                "unconditional": self.n_epochs_unc,
                "moment": self.n_epochs_moment,
                "conditional": self.n_epochs_cond,
            }
            if phase not in phase_totals:
                raise ValueError(f"default_checkpoint phase is invalid; got {phase!r}")
            _require_int("default_checkpoint epoch", checkpoint)
            if checkpoint > phase_totals[phase]:
                raise ValueError(
                    f"default_checkpoint epoch must be <= {phase_totals[phase]}; got {checkpoint}"
                )
